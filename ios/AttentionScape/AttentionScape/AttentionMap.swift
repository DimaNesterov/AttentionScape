import CoreGraphics
import Foundation
import RealityKit
import simd
import UIKit

/// Ошибки загрузки карты внимания.
enum AttentionError: LocalizedError {
    case missingFiles
    case noSessionID
    case unknownSession(String)
    case meshMismatch
    case badSize
    case empty

    var errorDescription: String? {
        switch self {
        case .missingFiles: return "Pick both attention.json and attention_f32.bin (or the results folder)"
        case .noSessionID: return "attention.json has no session_id"
        case .unknownSession(let id): return "Session \(id) is not on this iPhone"
        case .meshMismatch: return "The map was computed for a different scan (mesh_sha256 mismatch)"
        case .badSize: return "attention_f32.bin has an unexpected size"
        case .empty: return "The map has no painted surfaces"
        }
    }
}

/// results/attention.json (docs/data_spec.md, раздел 4.4) — только нужные поля.
struct AttentionMeta: Codable {
    var session_id: String?
    var mesh_sha256: String?
    var n_vertices: Int
    var n_time_bins: Int
}

private struct SessionHashHeader: Decodable {
    let mesh_sha256: String?
}

/// Внимание по вершинам меша (сумма по временным интервалам) и покрытие.
struct AttentionResults {
    let meta: AttentionMeta
    let attention: [Float]
    let coverage: [Float]

    /// nil — у сессии нет карты внимания.
    static func load(sessionDir: URL) throws -> AttentionResults? {
        let dir = sessionDir.appendingPathComponent("results", isDirectory: true)
        let jsonURL = dir.appendingPathComponent("attention.json")
        let binURL = dir.appendingPathComponent("attention_f32.bin")
        let fm = FileManager.default
        guard fm.fileExists(atPath: jsonURL.path), fm.fileExists(atPath: binURL.path) else { return nil }

        let meta = try JSONDecoder().decode(AttentionMeta.self, from: Data(contentsOf: jsonURL))
        let data = try Data(contentsOf: binURL)
        let n = meta.n_vertices
        let bins = max(meta.n_time_bins, 1)
        guard data.count >= (bins + 1) * n * 4 else { throw AttentionError.badSize }

        // float32 little-endian: attention[t][v], затем coverage[v].
        var floats = [Float](repeating: 0, count: (bins + 1) * n)
        _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        var attention = [Float](repeating: 0, count: n)
        for t in 0..<bins {
            for v in 0..<n {
                attention[v] += floats[t * n + v]
            }
        }
        let coverage = Array(floats[(bins * n)..<((bins + 1) * n)])
        return AttentionResults(meta: meta, attention: attention, coverage: coverage)
    }
}

/// Карта внимания: цветной слой на поверхностях скана.
/// В AR к нему добавляется невидимый «окклюдер» из скана, который прячет пятна за предметами.
@MainActor
enum AttentionMap {

    enum Style {
        case ar          // поверх реальной комнаты: полупрозрачно + окклюдер
        case dollhouse   // 3D-вид: непрозрачно, дальние от зрителя стороны стен не рисуются
    }

    struct Overlay {
        let heat: ModelEntity
        let occluder: ModelEntity?
        let paintedFraction: Float
    }

    /// Как в pipeline/tools/view_attention.py: слабее этого уровня не красим.
    static let minLevel: Float = 0.06
    /// Цветной слой чуть впереди поверхности, окклюдер чуть позади — чтобы не мерцали.
    static let heatOffset: Float = 0.012
    static let occluderInset: Float = 0.01
    static let heatName = "attentionHeat"

    /// Шкала синий → голубой → зелёный → жёлтый → оранжевый → красный (та же, что на Mac).
    static let stops: [Float] = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0]
    static let colors: [SIMD3<Float>] = [
        SIMD3(35, 23, 125), SIMD3(40, 90, 220), SIMD3(30, 170, 230), SIMD3(40, 210, 140),
        SIMD3(150, 225, 50), SIMD3(250, 200, 40), SIMD3(245, 110, 30), SIMD3(210, 25, 25),
    ]

    static func color(at t: Float) -> SIMD3<Float> {
        let x = min(max(t, 0), 1)
        for i in 0..<(stops.count - 1) where x <= stops[i + 1] {
            let f = (x - stops[i]) / (stops[i + 1] - stops[i])
            return colors[i] + (colors[i + 1] - colors[i]) * f
        }
        return colors[colors.count - 1]
    }

    // MARK: - Импорт результатов с Mac

    /// Принимает выбранные в Files файлы (attention.json и attention_f32.bin) или папку results.
    /// Проверяет, что карта посчитана для скана этой сессии, и копирует её в sessions/<id>/results/.
    /// Возвращает id сессии.
    static func importResults(from urls: [URL]) throws -> String {
        var jsonData: Data?
        var binData: Data?
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            let candidates = isDirectory
                ? [url.appendingPathComponent("attention.json"), url.appendingPathComponent("attention_f32.bin")]
                : [url]
            for file in candidates where file.lastPathComponent.hasPrefix("attention") {
                switch file.pathExtension.lowercased() {
                case "json": jsonData = try coordinatedRead(file)
                case "bin": binData = try coordinatedRead(file)
                default: continue
                }
            }
        }
        guard let jsonData, let binData else { throw AttentionError.missingFiles }

        let meta = try JSONDecoder().decode(AttentionMeta.self, from: jsonData)
        guard let id = meta.session_id else { throw AttentionError.noSessionID }
        let sessionDir = SessionFiles.sessionsRoot.appendingPathComponent(id, isDirectory: true)
        guard SessionFiles.isComplete(sessionDir) else { throw AttentionError.unknownSession(id) }
        let header = try JSONDecoder().decode(
            SessionHashHeader.self, from: Data(contentsOf: sessionDir.appendingPathComponent("session.json")))
        guard header.mesh_sha256 == meta.mesh_sha256 else { throw AttentionError.meshMismatch }
        guard binData.count == (max(meta.n_time_bins, 1) + 1) * meta.n_vertices * 4 else { throw AttentionError.badSize }

        let resultsDir = sessionDir.appendingPathComponent("results", isDirectory: true)
        try FileManager.default.createDirectory(at: resultsDir, withIntermediateDirectories: true)
        try jsonData.write(to: resultsDir.appendingPathComponent("attention.json"), options: .atomic)
        try binData.write(to: resultsDir.appendingPathComponent("attention_f32.bin"), options: .atomic)
        return id
    }

    /// Чтение через NSFileCoordinator: если файл в iCloud ещё не скачан, он скачается.
    private static func coordinatedRead(_ url: URL) throws -> Data {
        var coordinatorError: NSError?
        var result: Result<Data, Error>?
        NSFileCoordinator().coordinate(readingItemAt: url, options: [], error: &coordinatorError) { readURL in
            result = Result { try Data(contentsOf: readURL) }
        }
        if let coordinatorError { throw coordinatorError }
        guard let result else { throw CocoaError(.fileReadUnknown) }
        return try result.get()
    }

    // MARK: - Построение слоя

    /// Уровень 0…1 по вершинам: нормировка по 99,5-му перцентилю и корень — как на Mac.
    static func levels(_ attention: [Float]) -> [Float] {
        let positive = attention.filter { $0 > 0 }.sorted()
        guard !positive.isEmpty else { return [Float](repeating: 0, count: attention.count) }
        let index = Int((Float(positive.count - 1) * 0.995).rounded())
        let scale = max(positive[index], 1e-12)
        return attention.map { $0 > 0 ? min($0 / scale, 1).squareRoot() : 0 }
    }

    /// Нормали вершин, направленные «наружу» (в свободное пространство), и знак обхода треугольников.
    /// Порядок обхода в скане ARKit согласован, поэтому знак определяется один раз:
    /// нормали пола должны смотреть вверх (если пола нет — нормали потолка вниз).
    /// sign = −1 означает, что треугольники нужно перевернуть, чтобы их лицевая сторона смотрела в комнату.
    static func orientation(_ mesh: WorldMesh) -> (normals: [SIMD3<Float>], sign: Float) {
        var floorSum: Float = 0
        var ceilingSum: Float = 0
        for f in 0..<mesh.faceCount where mesh.classifications[f] == 2 || mesh.classifications[f] == 3 {
            let a = mesh.positions[Int(mesh.indices[f * 3])]
            let b = mesh.positions[Int(mesh.indices[f * 3 + 1])]
            let c = mesh.positions[Int(mesh.indices[f * 3 + 2])]
            let ny = simd_cross(b - a, c - a).y
            if mesh.classifications[f] == 2 { floorSum += ny } else { ceilingSum -= ny }
        }
        let evidence = floorSum != 0 ? floorSum : ceilingSum
        let sign: Float = evidence < 0 ? -1 : 1
        return (MeshIO.vertexNormals(mesh).map { $0 * sign }, sign)
    }

    /// Индексы треугольников с лицевой стороной в сторону комнаты (обход против часовой стрелки).
    static func inwardIndices(_ indices: [UInt32], sign: Float) -> [UInt32] {
        guard sign < 0 else { return indices }
        var flipped = indices
        for f in stride(from: 0, to: flipped.count - 2, by: 3) {
            flipped.swapAt(f + 1, f + 2)
        }
        return flipped
    }

    /// Текстура-шкала 256×4: координата u вершины = её уровень внимания.
    static func colormapImage(width: Int = 256, height: Int = 4) -> CGImage? {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for x in 0..<width {
            let rgb = color(at: Float(x) / Float(width - 1))
            for y in 0..<height {
                let i = (y * width + x) * 4
                pixels[i] = UInt8(min(max(rgb.x, 0), 255))
                pixels[i + 1] = UInt8(min(max(rgb.y, 0), 255))
                pixels[i + 2] = UInt8(min(max(rgb.z, 0), 255))
            }
        }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return pixels.withUnsafeMutableBytes { raw -> CGImage? in
            CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: width * 4, space: space,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        }
    }

    private static func textureU(_ level: Float) -> Float {
        let t = min(max((level - minLevel) / (1 - minLevel), 0), 1)
        return 0.5 / 256 + t * (255.0 / 256)        // не выходим на край текстуры
    }

    static func makeOverlay(mesh: WorldMesh, results: AttentionResults,
                            style: Style, opacity: Float = 1) async throws -> Overlay {
        guard results.attention.count == mesh.positions.count else { throw AttentionError.meshMismatch }
        let level = levels(results.attention)
        let (normals, sign) = orientation(mesh)
        let faces = style == .dollhouse ? inwardIndices(mesh.indices, sign: sign) : mesh.indices

        // Цветной слой: только треугольники, где хотя бы одна вершина выше порога.
        var remap = [Int32](repeating: -1, count: mesh.positions.count)
        var positions: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        var indices: [UInt32] = []
        for f in 0..<mesh.faceCount {
            let tri = [Int(faces[f * 3]), Int(faces[f * 3 + 1]), Int(faces[f * 3 + 2])]
            guard max(level[tri[0]], level[tri[1]], level[tri[2]]) >= minLevel else { continue }
            for v in tri {
                if remap[v] < 0 {
                    remap[v] = Int32(positions.count)
                    positions.append(mesh.positions[v] + normals[v] * heatOffset)
                    uvs.append(SIMD2<Float>(textureU(level[v]), 0.5))
                }
                indices.append(UInt32(remap[v]))
            }
        }
        guard !indices.isEmpty else { throw AttentionError.empty }

        var heatDescriptor = MeshDescriptor(name: heatName)
        heatDescriptor.positions = MeshBuffers.Positions(positions)
        heatDescriptor.textureCoordinates = MeshBuffers.TextureCoordinates(uvs)
        heatDescriptor.primitives = .triangles(indices)
        let heatMesh = try MeshResource.generate(from: [heatDescriptor])

        guard let image = colormapImage() else { throw AttentionError.empty }
        let texture = try await TextureResource(
            image: image, withName: "attentionColormap",
            options: TextureResource.CreateOptions(semantic: .color, mipmapsMode: .none))
        var material = UnlitMaterial(applyPostProcessToneMap: false)
        material.color = .init(tint: .white, texture: MaterialParameters.Texture(texture))
        switch style {
        case .ar:
            material.blending = .transparent(opacity: .init(scale: opacity))
            material.faceCulling = .none
        case .dollhouse:
            material.blending = .opaque
            material.faceCulling = .back
        }
        let heat = ModelEntity(mesh: heatMesh, materials: [material])
        heat.name = heatName

        var occluder: ModelEntity?
        if style == .ar {
            // Окклюдер: весь скан, чуть утопленный в поверхности; сам невидим, но закрывает то, что за ним.
            var occluderDescriptor = MeshDescriptor(name: "attentionOccluder")
            occluderDescriptor.positions = MeshBuffers.Positions(
                zip(mesh.positions, normals).map { $0 - $1 * occluderInset })
            occluderDescriptor.primitives = .triangles(mesh.indices)
            occluder = ModelEntity(mesh: try MeshResource.generate(from: [occluderDescriptor]),
                                   materials: [OcclusionMaterial()])
        }

        return Overlay(heat: heat, occluder: occluder,
                       paintedFraction: Float(indices.count / 3) / Float(max(mesh.faceCount, 1)))
    }

    /// Меняет прозрачность уже построенного слоя (без пересборки).
    static func setOpacity(_ heat: ModelEntity, _ opacity: Float) {
        guard var material = heat.model?.materials.first as? UnlitMaterial else { return }
        material.blending = .transparent(opacity: .init(scale: opacity))
        heat.model?.materials = [material]
    }
}
