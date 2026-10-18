import Foundation
import RealityKit
import simd
import SwiftUI
import UIKit

/// 3D-вид сессии «кукольный домик»: скан + карта внимания + места камеры + подписи комнат.
/// Работает без привязки к комнате. Ближние к зрителю стены не рисуются (видна их обратная сторона),
/// поэтому комнату видно снаружи, сверху и сбоку, включая стороны предметов, скрытые от камеры.
/// Жесты: один палец — вращать, два пальца — сдвигать, щипок — приближать, двойной тап — исходный вид.
struct DollhouseView: View {
    let session: SessionSummary
    @Environment(\.dismiss) private var dismiss
    @State private var info = "Loading…"
    @State private var hasMap = false
    @State private var showMap = true

    var body: some View {
        ZStack(alignment: .top) {
            DollhouseContainer(sessionDir: session.url, showMap: showMap) { text, mapLoaded in
                info = text
                hasMap = mapLoaded
            }
            .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(session.id)
                        .font(.subheadline.monospaced())
                    Spacer()
                    Button("Done") { dismiss() }
                }
                Text(info)
                    .font(.caption)
                Text("1 finger: rotate · 2 fingers: move · pinch: zoom · double tap: reset")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if hasMap {
                    if showMap {
                        AttentionLegend()
                    }
                    Button(showMap ? "Show scan" : "Show map") { showMap.toggle() }
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .padding(12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            .padding()
        }
    }
}

/// ARView без AR со своей камерой (PerspectiveCamera) и своими жестами.
struct DollhouseContainer: UIViewRepresentable {
    let sessionDir: URL
    let showMap: Bool
    let onLoaded: (String, Bool) -> Void

    func makeCoordinator() -> OrbitCamera { OrbitCamera() }

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
        view.environment.background = .color(UIColor(white: 0.12, alpha: 1))
        context.coordinator.attach(to: view)
        let coordinator = context.coordinator
        let onLoaded = onLoaded
        Task {
            do {
                let scene = try await DollhouseScene.build(sessionDir: sessionDir)
                coordinator.setContent(scene.root, radius: scene.radius)
                onLoaded(scene.info, scene.hasMap)
            } catch {
                onLoaded("Could not load: \(error.localizedDescription)", false)
            }
        }
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.content?.findEntity(named: AttentionMap.heatName)?.isEnabled = showMap
    }
}

/// Камера, вращающаяся вокруг точки target: угол по горизонтали (yaw), наклон (pitch), расстояние.
final class OrbitCamera: NSObject, UIGestureRecognizerDelegate {
    private let camera = PerspectiveCamera()
    private let anchor = AnchorEntity(world: .zero)
    private weak var view: ARView?
    private(set) var content: Entity?

    private var target = SIMD3<Float>(repeating: 0)
    private var yaw: Float = 0.6
    private var pitch: Float = 0.9           // радианы над горизонтом: 0.9 ≈ 50°, смотрим сверху-сбоку
    private var distance: Float = 10
    private var home: (SIMD3<Float>, Float, Float, Float) = (.zero, 0.6, 0.9, 10)
    private let fov: Float = 60

    func attach(to view: ARView) {
        self.view = view
        camera.camera.fieldOfViewInDegrees = fov
        anchor.addChild(camera)
        view.scene.addAnchor(anchor)

        let rotate = UIPanGestureRecognizer(target: self, action: #selector(handleRotate(_:)))
        rotate.minimumNumberOfTouches = 1
        rotate.maximumNumberOfTouches = 1
        let move = UIPanGestureRecognizer(target: self, action: #selector(handleMove(_:)))
        move.minimumNumberOfTouches = 2
        move.maximumNumberOfTouches = 2
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        let reset = UITapGestureRecognizer(target: self, action: #selector(handleReset(_:)))
        reset.numberOfTapsRequired = 2
        for gesture in [rotate, move, pinch, reset] as [UIGestureRecognizer] {
            gesture.delegate = self
            view.addGestureRecognizer(gesture)
        }
        updateCamera()
    }

    /// Сцена центрирована в начале координат; radius — половина наибольшего размера скана.
    func setContent(_ root: Entity, radius: Float) {
        content?.removeFromParent()
        content = root
        anchor.addChild(root)
        distance = max(radius * 2.2, 2)
        home = (target, yaw, pitch, distance)
        updateCamera()
    }

    private func updateCamera() {
        let offset = SIMD3<Float>(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)) * distance
        camera.look(at: target, from: target + offset, relativeTo: nil)
    }

    /// Сколько метров сцены приходится на 1 точку экрана на расстоянии target.
    private var metersPerPoint: Float {
        let height = Float(max(view?.bounds.height ?? 800, 1))
        return 2 * distance * tan(fov * .pi / 360) / height
    }

    @objc private func handleRotate(_ g: UIPanGestureRecognizer) {
        let t = g.translation(in: g.view)
        g.setTranslation(.zero, in: g.view)
        yaw -= Float(t.x) * 0.006
        pitch = min(max(pitch + Float(t.y) * 0.006, -0.3), 1.55)
        updateCamera()
    }

    @objc private func handleMove(_ g: UIPanGestureRecognizer) {
        let t = g.translation(in: g.view)
        g.setTranslation(.zero, in: g.view)
        // Оси экрана в мире: вправо и вверх относительно взгляда камеры.
        let forward = simd_normalize(-SIMD3<Float>(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)))
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        let up = simd_cross(right, forward)
        target += (-right * Float(t.x) + up * Float(t.y)) * metersPerPoint
        updateCamera()
    }

    @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
        distance = min(max(distance / Float(g.scale), 0.3), 80)
        g.scale = 1
        updateCamera()
    }

    @objc private func handleReset(_ g: UITapGestureRecognizer) {
        (target, yaw, pitch, distance) = home
        updateCamera()
    }

    /// Щипок и сдвиг двумя пальцами работают одновременно.
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        (gestureRecognizer is UIPinchGestureRecognizer && other is UIPanGestureRecognizer)
            || (gestureRecognizer is UIPanGestureRecognizer && other is UIPinchGestureRecognizer)
    }
}

/// Сборка сцены для 3D-вида.
@MainActor
enum DollhouseScene {

    struct Built {
        let root: Entity
        let hasMap: Bool
        let info: String
        let radius: Float      // половина наибольшего размера скана (м) — для начального расстояния камеры
    }

    static func build(sessionDir: URL) async throws -> Built {
        let scanDir = sessionDir.appendingPathComponent("scan", isDirectory: true)
        let mesh = try MeshIO.readPLY(Data(contentsOf: scanDir.appendingPathComponent("mesh.ply")))
        guard mesh.faceCount > 0 else { throw MeshIOError.truncated }
        let (normals, sign) = AttentionMap.orientation(mesh)

        // Комната ставится центром в начало координат — вокруг него вращается камера.
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in mesh.positions {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        let root = Entity()
        let room = Entity()
        room.position = -(lo + hi) / 2
        root.addChild(room)

        room.addChild(try baseModel(mesh, normals: normals, sign: sign))

        var info = "\(mesh.positions.count) vertices · no attention map yet"
        var hasMap = false
        if let results = try AttentionResults.load(sessionDir: sessionDir) {
            let overlay = try await AttentionMap.makeOverlay(mesh: mesh, results: results, style: .dollhouse)
            room.addChild(overlay.heat)
            hasMap = true
            info = "Attention map: \(Int((overlay.paintedFraction * 100).rounded()))% of surfaces painted"
        }
        let cameras = cameraMarkers(sessionDir: sessionDir)
        for camera in cameras {
            room.addChild(camera)
        }
        if !cameras.isEmpty {
            info += cameras.count == 1 ? " · white box = camera" : " · white boxes = \(cameras.count) camera spots"
        }
        let marks = Rooms.load(sessionDir: sessionDir)
        if !marks.isEmpty {
            let shares = Rooms.shares(sessionDir: sessionDir)
            let floorPoints = floorVertices(mesh)
            for mark in marks {
                let share = shares[mark.name].map { " \(Int(($0 * 100).rounded()))%" } ?? ""
                // Высота пола берётся под самой отметкой: в разных комнатах пол в скане бывает на разной высоте.
                let floorY = floorHeight(floorPoints, near: mark.point, fallback: mesh)
                room.addChild(roomLabel(mark.name + share, at: SIMD3(mark.point.x, floorY + 0.08, mark.point.z)))
            }
        }
        addLights(to: root)
        let size = hi - lo
        return Built(root: root, hasMap: hasMap, info: info, radius: max(size.x, size.y, size.z) / 2)
    }

    /// Скан светло-серым (пол темнее), только лицевые стороны — обращённые в комнату.
    private static func baseModel(_ mesh: WorldMesh, normals: [SIMD3<Float>], sign: Float) throws -> ModelEntity {
        var descriptor = MeshDescriptor(name: "dollhouseScan")
        descriptor.positions = MeshBuffers.Positions(mesh.positions)
        descriptor.normals = MeshBuffers.Normals(normals)
        descriptor.primitives = .triangles(AttentionMap.inwardIndices(mesh.indices, sign: sign))
        descriptor.materials = .perFace(mesh.classifications.map { $0 == 2 ? UInt32(1) : UInt32(0) })
        let resource = try MeshResource.generate(from: [descriptor])
        return ModelEntity(mesh: resource, materials: [surface(white: 0.82), surface(white: 0.6)])
    }

    private static func surface(white: CGFloat) -> PhysicallyBasedMaterial {
        var material = PhysicallyBasedMaterial()
        material.baseColor = .init(tint: UIColor(white: white, alpha: 1))
        material.roughness = 1.0
        material.metallic = 0.0
        material.faceCulling = .back
        return material
    }

    /// Белый «телефон» на каждом месте штатива: поза первого кадра каждой записи; тёмный блок — объектив.
    private static func cameraMarkers(sessionDir: URL) -> [Entity] {
        var firstOfTake: [FrameMeta] = []
        var lastTake: Int?
        var lastTime: Double?
        for meta in SessionFiles.loadFrameMetas(sessionDir: sessionDir) where meta.camera.T_world_from_camera_ar.count == 16 {
            // Новая запись: другой номер take или (у старых кадров без take) пауза больше 5 с.
            let isNew: Bool
            if let take = meta.take {
                isNew = take != lastTake
            } else {
                isNew = lastTime.map { meta.timestamp - $0 > 5 || meta.timestamp < $0 } ?? true
            }
            if isNew { firstOfTake.append(meta) }
            lastTake = meta.take
            lastTime = meta.timestamp
        }
        return firstOfTake.map { cameraMarker($0.camera.T_world_from_camera_ar) }
    }

    private static func cameraMarker(_ m: [Float]) -> Entity {
        // m — 4×4 построчно
        let matrix = simd_float4x4(columns: (
            SIMD4<Float>(m[0], m[4], m[8], m[12]),
            SIMD4<Float>(m[1], m[5], m[9], m[13]),
            SIMD4<Float>(m[2], m[6], m[10], m[14]),
            SIMD4<Float>(m[3], m[7], m[11], m[15])))
        let marker = Entity()
        marker.transform = Transform(matrix: matrix)

        let phone = ModelEntity(mesh: .generateBox(width: 0.16, height: 0.08, depth: 0.02, cornerRadius: 0.008),
                                materials: [UnlitMaterial(color: .white)])
        let lens = ModelEntity(mesh: .generateBox(size: [0.03, 0.03, 0.04]),
                               materials: [UnlitMaterial(color: .darkGray)])
        lens.position = [0, 0, -0.03]                      // камера ARKit смотрит вдоль −Z
        marker.addChild(phone)
        marker.addChild(lens)
        return marker
    }

    /// Вершины треугольников класса «пол».
    private static func floorVertices(_ mesh: WorldMesh) -> [SIMD3<Float>] {
        var points: [SIMD3<Float>] = []
        for f in 0..<mesh.faceCount where mesh.classifications[f] == 2 {
            points.append(mesh.positions[Int(mesh.indices[f * 3])])
        }
        return points
    }

    /// Высота пола рядом с точкой: медиана высот пола в радиусе 1 м (по плану),
    /// если там мало пола — в радиусе 2,5 м, иначе — по всему скану (или самая низкая точка скана).
    private static func floorHeight(_ floor: [SIMD3<Float>], near p: SIMD3<Float>, fallback mesh: WorldMesh) -> Float {
        for radius: Float in [1.0, 2.5, .greatestFiniteMagnitude] {
            var heights = floor.compactMap { q -> Float? in
                let dx = q.x - p.x, dz = q.z - p.z
                return dx * dx + dz * dz <= radius * radius ? q.y : nil
            }
            if heights.count >= 20 {
                heights.sort()
                return heights[heights.count / 2]
            }
        }
        return mesh.positions.map(\.y).min() ?? 0
    }

    /// Название комнаты чёрным, лежащее на полу (читается при взгляде сверху, как подпись на плане).
    private static func roomLabel(_ text: String, at position: SIMD3<Float>) -> Entity {
        let mesh = MeshResource.generateText(text, extrusionDepth: 0.01,
                                             font: .systemFont(ofSize: 0.22, weight: .bold))
        let label = ModelEntity(mesh: mesh, materials: [UnlitMaterial(color: .black)])
        label.position = -mesh.bounds.center                // центр текста — в точке отметки
        let holder = Entity()
        holder.position = position
        holder.orientation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])   // текст ложится на пол лицом вверх
        holder.addChild(label)
        return holder
    }

    /// Два направленных источника с разных сторон, чтобы все стены были различимы.
    private static func addLights(to root: Entity) {
        let key = DirectionalLight()
        key.light.intensity = 4000
        key.look(at: .zero, from: [1.5, 4, 2.5], relativeTo: nil)
        let fill = DirectionalLight()
        fill.light.intensity = 1500
        fill.look(at: .zero, from: [-2, 1.5, -3], relativeTo: nil)
        root.addChild(key)
        root.addChild(fill)
    }
}
