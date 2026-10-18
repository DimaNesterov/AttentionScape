import Foundation
import simd

/// Результат подгонки живой геометрии лидара к сохранённому скану.
nonisolated struct AlignmentResult: Sendable {
    /// Поправка: точка в текущих координатах сессии → координаты сохранённого скана.
    let transform: simd_float4x4
    /// Найденный и исправленный сдвиг (м) и поворот вокруг вертикали (°).
    let shift: Float
    let yawDegrees: Float
    /// Остаток после исправления: среднеквадратичное расстояние совпавших точек до скана (м).
    let fitRMS: Float
    /// Доля точек лидара, которые легли на скан (остальное — то, чего в скане нет: люди, переставленные вещи).
    let matchedFraction: Float

    var isReliable: Bool { matchedFraction >= 0.5 && fitRMS <= 0.03 }
}

/// Подгонка ICP «точка → плоскость» с 4 степенями свободы: сдвиг по x, y, z и поворот вокруг вертикали.
/// Наклон не подгоняется: и скан, и живая сессия ARKit уже выровнены по гравитации.
/// Точки дальше порога от скана не участвуют, поэтому изменения в комнате не сбивают подгонку.
nonisolated enum Aligner {

    private struct Stage {
        let cell: Float        // размер ячейки сетки поиска соседей и порог совпадения (м)
        let iterations: Int
    }

    /// От грубого к точному: сначала ловим большой сдвиг, потом уточняем.
    private static let stages = [
        Stage(cell: 0.40, iterations: 10),
        Stage(cell: 0.15, iterations: 10),
        Stage(cell: 0.05, iterations: 15),
    ]

    static func align(source: [SIMD3<Float>],
                      targetPoints: [SIMD3<Float>],
                      targetNormals: [SIMD3<Float>]) -> AlignmentResult? {
        guard source.count >= 200, targetPoints.count == targetNormals.count, !targetPoints.isEmpty else { return nil }
        var yaw: Float = 0
        var t = SIMD3<Float>(repeating: 0)

        for stage in stages {
            let grid = Grid(points: targetPoints, normals: targetNormals, cell: stage.cell)
            for _ in 0..<stage.iterations {
                var ata = simd_double4x4()
                var atb = SIMD4<Double>(repeating: 0)
                var count = 0
                for p in source {
                    let moved = rotateY(p, yaw) + t
                    guard let (q, n) = grid.nearest(to: moved, maxDistance: stage.cell) else { continue }
                    let r = Double(simd_dot(n, moved - q))
                    let dRot = SIMD3<Float>(moved.z, 0, -moved.x)          // производная поворота вокруг Y
                    let j = SIMD4<Double>(Double(simd_dot(n, dRot)), Double(n.x), Double(n.y), Double(n.z))
                    ata += simd_double4x4(columns: (j * j.x, j * j.y, j * j.z, j * j.w))
                    atb -= j * r
                    count += 1
                }
                guard count >= 50 else { return nil }
                let damping = 1e-4 * Double(count)                       // устойчивость, если видна одна стена
                ata += simd_double4x4(diagonal: SIMD4<Double>(repeating: damping))
                let delta = simd_inverse(ata) * atb
                let dYaw = Float(delta.x)
                yaw += dYaw
                t = rotateY(t, dYaw) + SIMD3<Float>(Float(delta.y), Float(delta.z), Float(delta.w))
                if abs(delta.x) < 1e-5 && simd_length(SIMD3<Double>(delta.y, delta.z, delta.w)) < 1e-4 { break }
            }
        }

        // Качество: сколько точек легло на скан (в пределах 5 см) и насколько плотно.
        let grid = Grid(points: targetPoints, normals: targetNormals, cell: 0.05)
        var sumSquares: Float = 0
        var matched = 0
        for p in source {
            let moved = rotateY(p, yaw) + t
            guard let (q, n) = grid.nearest(to: moved, maxDistance: 0.05) else { continue }
            let r = simd_dot(n, moved - q)
            sumSquares += r * r
            matched += 1
        }
        guard matched > 0 else { return nil }

        let c = cos(yaw), s = sin(yaw)
        let transform = simd_float4x4(columns: (
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(s, 0, c, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)))
        return AlignmentResult(transform: transform,
                               shift: simd_length(t),
                               yawDegrees: yaw * 180 / .pi,
                               fitRMS: (sumSquares / Float(matched)).squareRoot(),
                               matchedFraction: Float(matched) / Float(source.count))
    }

    /// Поворот вокруг вертикальной оси Y (та же матрица, что в transform).
    private static func rotateY(_ p: SIMD3<Float>, _ a: Float) -> SIMD3<Float> {
        let c = cos(a), s = sin(a)
        return SIMD3<Float>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z)
    }

    /// Равномерно прореживает точки до maxCount.
    static func subsample(_ points: [SIMD3<Float>], maxCount: Int) -> [SIMD3<Float>] {
        guard points.count > maxCount else { return points }
        let step = Double(points.count) / Double(maxCount)
        return (0..<maxCount).map { points[Int(Double($0) * step)] }
    }

    /// Сетка для поиска ближайшей точки скана. Скан прореживается до 1 точки на (cell/3)³,
    /// чтобы поиск оставался быстрым и на грубых шагах.
    private struct Grid {
        let cell: Float
        var points: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var buckets: [SIMD3<Int32>: [Int32]] = [:]

        init(points source: [SIMD3<Float>], normals sourceNormals: [SIMD3<Float>], cell: Float) {
            self.cell = cell
            let voxel = cell / 3
            var seen = Set<SIMD3<Int32>>()
            for (p, n) in zip(source, sourceNormals) {
                guard seen.insert(Self.key(p, voxel)).inserted else { continue }
                buckets[Self.key(p, cell), default: []].append(Int32(points.count))
                points.append(p)
                normals.append(n)
            }
        }

        static func key(_ p: SIMD3<Float>, _ size: Float) -> SIMD3<Int32> {
            SIMD3<Int32>(Int32((p.x / size).rounded(.down)),
                         Int32((p.y / size).rounded(.down)),
                         Int32((p.z / size).rounded(.down)))
        }

        func nearest(to p: SIMD3<Float>, maxDistance: Float) -> (SIMD3<Float>, SIMD3<Float>)? {
            let center = Self.key(p, cell)
            var best: Int32 = -1
            var bestDistance = maxDistance * maxDistance
            for dx in Int32(-1)...1 {
                for dy in Int32(-1)...1 {
                    for dz in Int32(-1)...1 {
                        guard let bucket = buckets[center &+ SIMD3<Int32>(dx, dy, dz)] else { continue }
                        for i in bucket {
                            let d = simd_distance_squared(points[Int(i)], p)
                            if d < bestDistance {
                                bestDistance = d
                                best = i
                            }
                        }
                    }
                }
            }
            return best >= 0 ? (points[Int(best)], normals[Int(best)]) : nil
        }
    }
}
