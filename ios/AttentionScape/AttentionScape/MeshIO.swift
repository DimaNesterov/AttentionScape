import ARKit
import CryptoKit
import Foundation
import Metal
import RealityKit
import simd
import UIKit

/// Меш в мировой системе координат W (см. docs/data_spec.md).
struct WorldMesh {
    var positions: [SIMD3<Float>] = []
    var indices: [UInt32] = []          // по 3 индекса на треугольник
    var classifications: [UInt8] = []   // по 1 коду класса ARKit на треугольник
    var faceCount: Int { indices.count / 3 }
}

enum MeshIOError: Error {
    case badHeader
    case truncated
}

enum MeshIO {

    /// Цвета классов ARKit (0 none, 1 wall, 2 floor, 3 ceiling, 4 table, 5 seat, 6 window, 7 door).
    /// Те же цвета, что в pipeline/tools/view_mesh.py.
    static let classRGB: [SIMD3<Float>] = [
        SIMD3(150, 150, 150), SIMD3(205, 200, 190), SIMD3(170, 120, 80), SIMD3(240, 240, 240),
        SIMD3(60, 130, 220), SIMD3(60, 180, 90), SIMD3(120, 220, 230), SIMD3(220, 150, 60),
    ]

    /// Собирает все ARMeshAnchor в один меш в мировых координатах.
    static func collect(from anchors: [ARAnchor]) -> WorldMesh {
        var mesh = WorldMesh()
        for case let anchor as ARMeshAnchor in anchors {
            let geometry = anchor.geometry
            let base = UInt32(mesh.positions.count)
            let transform = anchor.transform

            let vertices = geometry.vertices
            let vertexPointer = vertices.buffer.contents()
            for i in 0..<vertices.count {
                let p = vertexPointer
                    .advanced(by: vertices.offset + vertices.stride * i)
                    .assumingMemoryBound(to: Float.self)
                let world = transform * SIMD4<Float>(p[0], p[1], p[2], 1)
                mesh.positions.append(SIMD3<Float>(world.x, world.y, world.z))
            }

            let faces = geometry.faces
            let facePointer = faces.buffer.contents()
            let classification = geometry.classification
            for f in 0..<faces.count {
                for k in 0..<3 {
                    let byteOffset = (f * faces.indexCountPerPrimitive + k) * faces.bytesPerIndex
                    let index: UInt32
                    if faces.bytesPerIndex == 2 {
                        index = UInt32(facePointer.load(fromByteOffset: byteOffset, as: UInt16.self))
                    } else {
                        index = facePointer.load(fromByteOffset: byteOffset, as: UInt32.self)
                    }
                    mesh.indices.append(base + index)
                }
                var cls: UInt8 = 0
                if let classification {
                    cls = classification.buffer.contents()
                        .load(fromByteOffset: classification.offset + classification.stride * f, as: UInt8.self)
                }
                mesh.classifications.append(cls)
            }
        }
        return mesh
    }

    // MARK: - PLY (бинарный, little-endian), формат из data_spec.md

    static func plyData(_ mesh: WorldMesh) -> Data {
        var header = "ply\nformat binary_little_endian 1.0\n"
        header += "comment AttentionScape mesh, world frame W (ARKit), meters\n"
        header += "element vertex \(mesh.positions.count)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "element face \(mesh.faceCount)\n"
        header += "property list uchar uint vertex_indices\nproperty uchar classification\n"
        header += "end_header\n"

        var data = Data(header.utf8)
        data.reserveCapacity(data.count + mesh.positions.count * 12 + mesh.faceCount * 14)
        for p in mesh.positions {
            append(p.x.bitPattern, to: &data)
            append(p.y.bitPattern, to: &data)
            append(p.z.bitPattern, to: &data)
        }
        for f in 0..<mesh.faceCount {
            data.append(UInt8(3))
            append(mesh.indices[f * 3], to: &data)
            append(mesh.indices[f * 3 + 1], to: &data)
            append(mesh.indices[f * 3 + 2], to: &data)
            data.append(mesh.classifications[f])
        }
        return data
    }

    /// Читает PLY, записанный функцией plyData (только этот формат).
    static func readPLY(_ data: Data) throws -> WorldMesh {
        guard let headerEnd = data.range(of: Data("end_header\n".utf8)) else { throw MeshIOError.badHeader }
        let header = String(decoding: data[data.startIndex..<headerEnd.lowerBound], as: UTF8.self)
        var vertexCount = 0
        var faceCount = 0
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count == 3, parts[0] == "element" else { continue }
            if parts[1] == "vertex" { vertexCount = Int(parts[2]) ?? 0 }
            if parts[1] == "face" { faceCount = Int(parts[2]) ?? 0 }
        }

        var mesh = WorldMesh()
        mesh.positions.reserveCapacity(vertexCount)
        mesh.indices.reserveCapacity(faceCount * 3)
        mesh.classifications.reserveCapacity(faceCount)

        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var o = headerEnd.upperBound - data.startIndex
            guard raw.count >= o + vertexCount * 12 + faceCount * 14 else { throw MeshIOError.truncated }
            for _ in 0..<vertexCount {
                let x = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o, as: UInt32.self)))
                let y = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o + 4, as: UInt32.self)))
                let z = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o + 8, as: UInt32.self)))
                mesh.positions.append(SIMD3<Float>(x, y, z))
                o += 12
            }
            for _ in 0..<faceCount {
                o += 1 // число вершин в грани (всегда 3)
                for _ in 0..<3 {
                    mesh.indices.append(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o, as: UInt32.self)))
                    o += 4
                }
                mesh.classifications.append(raw[o])
                o += 1
            }
        }
        return mesh
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Нормали вершин (сумма нормалей соседних треугольников) — для освещения.
    static func vertexNormals(_ mesh: WorldMesh) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: mesh.positions.count)
        for f in 0..<mesh.faceCount {
            let a = Int(mesh.indices[f * 3])
            let b = Int(mesh.indices[f * 3 + 1])
            let c = Int(mesh.indices[f * 3 + 2])
            let n = simd_cross(mesh.positions[b] - mesh.positions[a], mesh.positions[c] - mesh.positions[a])
            normals[a] += n
            normals[b] += n
            normals[c] += n
        }
        return normals.map { simd_length($0) > 0 ? simd_normalize($0) : SIMD3<Float>(0, 1, 0) }
    }

    /// 3D-меш для отображения: освещение + цвет по классу поверхности (материал на каждый треугольник).
    @MainActor
    static func classColoredModel(_ mesh: WorldMesh, opacity: Float) throws -> ModelEntity {
        var descriptor = MeshDescriptor(name: "savedScan")
        descriptor.positions = MeshBuffers.Positions(mesh.positions)
        descriptor.normals = MeshBuffers.Normals(vertexNormals(mesh))
        descriptor.primitives = .triangles(mesh.indices)
        descriptor.materials = .perFace(mesh.classifications.map { UInt32(min($0, 7)) })
        let resource = try MeshResource.generate(from: [descriptor])

        let materials: [any RealityKit.Material] = classRGB.map { rgb in
            var material = PhysicallyBasedMaterial()
            material.baseColor = .init(tint: UIColor(red: CGFloat(rgb.x / 255), green: CGFloat(rgb.y / 255),
                                                     blue: CGFloat(rgb.z / 255), alpha: 1))
            material.roughness = 1.0
            material.metallic = 0.0
            material.blending = .transparent(opacity: .init(scale: opacity))
            return material
        }
        return ModelEntity(mesh: resource, materials: materials)
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
}
