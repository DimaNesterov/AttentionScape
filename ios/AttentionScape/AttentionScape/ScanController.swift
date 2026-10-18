import ARKit
import Combine
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import RealityKit
import SwiftUI
import UIKit

struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

/// Что показывать поверх комнаты после привязки.
enum OverlayKind: String {
    case attention = "Map"     // карта внимания
    case scan = "Scan"         // скан в цветах классов поверхностей
}

@MainActor
final class ScanController: NSObject, ObservableObject {

    enum Mode: String {
        case idle = "Ready"
        case scanning = "Scanning"
        case saving = "Saving…"
        case relocalizing = "Relocalizing…"
        case relocalized = "Relocalized"
    }

    @Published var mode: Mode = .idle
    @Published var status = "Tap Start scan"
    @Published var tracking = "—"
    @Published var mapping = "—"
    @Published var meshInfo = ""
    @Published var liveChunks = ""
    @Published var shareItem: ShareItem?
    @Published var isRecording = false
    @Published var recordedFrames = 0
    @Published var sessions: [SessionSummary] = []
    @Published var selectedSessionID: String?
    @Published var overlayKind: OverlayKind = .attention
    @Published var showingAttention = false
    @Published var isAligning = false
    /// Отметки комнат текущего скана (во время сканирования или после привязки).
    @Published var rooms: [RoomMark] = []
    /// Прозрачность карты внимания в AR (0 — не видно, 1 — сплошной цвет).
    @Published var heatOpacity: Float = 0.5 {
        didSet {
            if let heat { AttentionMap.setOpacity(heat, heatOpacity) }
        }
    }

    var selectedSession: SessionSummary? {
        sessions.first { $0.id == selectedSessionID }
    }

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)

    private var overlay: AnchorEntity?
    private var heat: ModelEntity?
    private var overlayBuildID = 0
    private var relocalizedMesh: WorldMesh?
    private var alignTask: Task<Void, Never>?
    private var pendingRoomPosition: SIMD3<Float>?
    private var currentTake = 1
    private var lastUIUpdate: TimeInterval = 0

    // Запись кадров (неделя 2)
    private var relocalizedSessionDir: URL?
    private var framesDir: URL?
    private var nextFrameIndex = 1
    private var recordTask: Task<Void, Never>?
    private var highResSupported = false
    private var highResInFlight = false
    private let maxFrameWidth = 4032

    // Подсказка, если привязка долго не удаётся
    private var relocalizeStartTime: TimeInterval?
    private var relocalizeHintShown = false

    override init() {
        super.init()
        arView.session.delegate = self
        SessionFiles.cleanupTemporaryZips()
        refreshSessions()
    }

    // MARK: - Sessions

    func refreshSessions() {
        sessions = SessionFiles.listCompleteSessions()
        if selectedSession == nil {
            selectedSessionID = sessions.first?.id
        }
    }

    func selectSession(_ id: String) {
        selectedSessionID = id
        status = "Selected \(id). Tap Relocalize."
    }

    func deleteSession(_ id: String) {
        guard let session = sessions.first(where: { $0.id == id }) else { return }
        if relocalizedSessionDir == session.url {
            stopRecordingIfNeeded()
            cancelAlignment()
            removeOverlay()
            relocalizedSessionDir = nil
            relocalizedMesh = nil
            rooms = []
            arView.session.pause()
            mode = .idle
        }
        try? FileManager.default.removeItem(at: session.url)
        if selectedSessionID == id { selectedSessionID = nil }
        refreshSessions()
        status = "Deleted \(id)"
    }

    // MARK: - Scanning

    func startScan() {
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) else {
            status = "This device does not support LiDAR scene reconstruction"
            return
        }
        stopRecordingIfNeeded()
        cancelAlignment()
        removeOverlay()
        relocalizedSessionDir = nil
        relocalizedMesh = nil
        rooms = []
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        config.environmentTexturing = .none
        arView.debugOptions = [.showSceneUnderstanding]
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        mode = .scanning
        meshInfo = ""
        status = "Walk slowly through the rooms. In each room tap Mark room. Save when Map shows \"mapped\"."
    }

    /// Порядок сохранения: mesh.ply → worldmap.arworldmap → session.json (последним).
    /// session.json появляется только у полностью сохранённой сессии.
    func saveScan() {
        guard mode == .scanning, let frame = arView.session.currentFrame else { return }
        let mesh = MeshIO.collect(from: frame.anchors)
        guard mesh.faceCount > 0 else {
            status = "No mesh yet — keep scanning"
            return
        }
        mode = .saving
        status = "Saving…"

        let sessionID = SessionFiles.newSessionID()
        let sessionDir: URL
        let info: SessionInfo
        do {
            let scanDir = try SessionFiles.makeScanDir(sessionID: sessionID)
            sessionDir = scanDir.deletingLastPathComponent()
            let ply = MeshIO.plyData(mesh)
            try ply.write(to: scanDir.appendingPathComponent("mesh.ply"), options: .atomic)
            info = SessionInfo(
                session_id: sessionID,
                created_utc: SessionFiles.isoNow(),
                device: SessionFiles.machineIdentifier(),
                ios_version: UIDevice.current.systemVersion,
                app_version: SessionFiles.appVersion(),
                mesh_sha256: MeshIO.sha256Hex(ply),
                n_vertices: mesh.positions.count,
                n_faces: mesh.faceCount)
        } catch {
            status = "Save failed: \(error.localizedDescription)"
            mode = .scanning
            return
        }

        let mapURL = sessionDir
            .appendingPathComponent("scan", isDirectory: true)
            .appendingPathComponent("worldmap.arworldmap")
        arView.session.getCurrentWorldMap { @Sendable [weak self] map, error in
            let failure: String?
            if let map {
                do {
                    let data = try NSKeyedArchiver.archivedData(withRootObject: map, requiringSecureCoding: true)
                    try data.write(to: mapURL, options: .atomic)
                    failure = nil
                } catch {
                    failure = "World map save failed: \(error.localizedDescription)"
                }
            } else {
                failure = "World map not ready (\(error?.localizedDescription ?? "unknown error")). Keep scanning, then Save again."
            }
            let target = self
            Task { @MainActor in
                target?.finishSave(info: info, sessionDir: sessionDir, failure: failure)
            }
        }
    }

    private func finishSave(info: SessionInfo, sessionDir: URL, failure: String?) {
        if let failure {
            try? FileManager.default.removeItem(at: sessionDir) // не оставляем неполные сессии
            status = failure
            mode = .scanning
            return
        }
        do {
            try SessionFiles.writeJSON(info, to: sessionDir.appendingPathComponent("session.json"))
        } catch {
            try? FileManager.default.removeItem(at: sessionDir)
            status = "Save failed: \(error.localizedDescription)"
            mode = .scanning
            return
        }
        var roomsNote = ""
        if !rooms.isEmpty {
            do {
                try Rooms.save(rooms, sessionDir: sessionDir)
                roomsNote = " \(rooms.count) room(s) marked."
            } catch {
                roomsNote = " Room marks were not saved: \(error.localizedDescription)"
            }
        }
        // Скан завершён: останавливаем сканирование, новый — только по кнопке Start scan.
        arView.session.pause()
        arView.debugOptions = []
        liveChunks = ""
        meshInfo = "\(info.n_vertices) vertices, \(info.n_faces) triangles"
        refreshSessions()
        selectedSessionID = info.session_id
        mode = .idle
        status = "Saved \(info.session_id).\(roomsNote) Tap Relocalize to check it, or Start scan for a new one."
    }

    // MARK: - Relocalization

    func relocalizeSelected() {
        guard let sessionDir = selectedSession?.url else {
            status = "No saved session selected"
            return
        }
        stopRecordingIfNeeded()
        let scanDir = sessionDir.appendingPathComponent("scan", isDirectory: true)
        do {
            let mapData = try Data(contentsOf: scanDir.appendingPathComponent("worldmap.arworldmap"))
            guard let worldMap = try NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: mapData) else {
                status = "Could not read the world map"
                return
            }
            let mesh = try MeshIO.readPLY(Data(contentsOf: scanDir.appendingPathComponent("mesh.ply")))
            removeOverlay()

            let config = ARWorldTrackingConfiguration()
            config.initialWorldMap = worldMap
            // Формат видео, который позволяет делать отдельные кадры в полном разрешении камеры.
            if let format = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
                config.videoFormat = format
                highResSupported = true
            } else {
                highResSupported = false
            }
            // Живая геометрия лидара — для точной подгонки к сохранённому скану после привязки.
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
                config.sceneReconstruction = .mesh
            }
            arView.debugOptions = []
            cancelAlignment()
            arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])

            relocalizedSessionDir = sessionDir
            relocalizedMesh = mesh
            rooms = Rooms.load(sessionDir: sessionDir)
            relocalizeStartTime = nil
            relocalizeHintShown = false
            mode = .relocalizing
            liveChunks = ""
            meshInfo = "\(sessionDir.lastPathComponent): \(mesh.positions.count) vertices"
            status = "Point the phone at the scanned area, ideally from where you started."
            if !SessionFiles.hasAttentionMap(sessionDir: sessionDir) {
                overlayKind = .scan
            }
            rebuildOverlay()
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
    }

    /// Строит слой поверх комнаты для привязанной сессии: карту внимания или скан.
    /// Строится асинхронно; если за это время запросили другой слой, результат отбрасывается.
    private func rebuildOverlay() {
        guard let mesh = relocalizedMesh, let sessionDir = relocalizedSessionDir else { return }
        removeOverlay()
        overlayBuildID += 1
        let buildID = overlayBuildID
        let kind = overlayKind
        Task { [weak self] in
            guard let self else { return }
            do {
                let anchor = AnchorEntity(world: .zero)
                var newHeat: ModelEntity?
                var info = ""
                if kind == .attention, let results = try AttentionResults.load(sessionDir: sessionDir) {
                    let built = try await AttentionMap.makeOverlay(mesh: mesh, results: results,
                                                                   style: .ar, opacity: self.heatOpacity)
                    anchor.addChild(built.heat)
                    if let occluder = built.occluder { anchor.addChild(occluder) }
                    newHeat = built.heat
                    info = "Attention map: \(Int((built.paintedFraction * 100).rounded()))% of surfaces painted"
                } else {
                    anchor.addChild(try MeshIO.classColoredModel(mesh, opacity: 0.7))
                }
                guard buildID == self.overlayBuildID else { return }
                if let newHeat { AttentionMap.setOpacity(newHeat, self.heatOpacity) }   // слайдер могли сдвинуть во время сборки
                anchor.isEnabled = self.mode == .relocalized
                self.arView.scene.addAnchor(anchor)
                self.overlay = anchor
                self.heat = newHeat
                self.showingAttention = newHeat != nil
                if !info.isEmpty { self.meshInfo = info }
            } catch {
                guard buildID == self.overlayBuildID else { return }
                self.status = "Overlay failed: \(error.localizedDescription)"
            }
        }
    }

    private func removeOverlay() {
        if let overlay {
            arView.scene.removeAnchor(overlay)
            self.overlay = nil
        }
        heat = nil
        showingAttention = false
    }

    // MARK: - Attention map

    /// Результат выбора файлов в Files: attention.json + attention_f32.bin (или папка results).
    func importAttention(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            status = "Import cancelled: \(error.localizedDescription)"
        case .success(let urls):
            do {
                let id = try AttentionMap.importResults(from: urls)
                refreshSessions()
                selectedSessionID = id
                overlayKind = .attention
                if relocalizedSessionDir?.lastPathComponent == id {
                    rebuildOverlay()
                    status = "Attention map loaded for \(id)."
                } else {
                    status = "Attention map imported for \(id). Tap Relocalize."
                }
            } catch {
                status = "Import failed: \(error.localizedDescription)"
            }
        }
    }

    /// Переключает слой: карта внимания ↔ скан.
    func toggleOverlayKind() {
        overlayKind = overlayKind == .attention ? .scan : .attention
        rebuildOverlay()
    }

    // MARK: - Rooms

    /// Отмечать комнату можно во время скана или после привязки (и подгонки), но не во время записи.
    var canMarkRoom: Bool {
        mode == .scanning || (mode == .relocalized && !isAligning && !isRecording)
    }

    /// Запоминает текущее место телефона; название комнаты придёт следом в saveRoom.
    func beginMarkRoom() -> Bool {
        guard canMarkRoom, let frame = arView.session.currentFrame,
              case .normal = frame.camera.trackingState else {
            status = "Tracking is not ready — try Mark room again in a moment"
            return false
        }
        let t = frame.camera.transform.columns.3
        pendingRoomPosition = SIMD3<Float>(t.x, t.y, t.z)
        return true
    }

    func saveRoom(name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let position = pendingRoomPosition, !trimmed.isEmpty else { return }
        pendingRoomPosition = nil
        rooms = Rooms.upsert(Rooms.make(name: trimmed, position: position), into: rooms)
        // После привязки скан уже сохранён — пишем сразу; во время скана отметки сохранятся вместе со сканом.
        if mode == .relocalized, let sessionDir = relocalizedSessionDir {
            do {
                try Rooms.save(rooms, sessionDir: sessionDir)
                refreshSessions()
            } catch {
                status = "Room mark not saved: \(error.localizedDescription)"
                return
            }
        }
        status = "Marked \(trimmed). Rooms: \(rooms.map(\.name).joined(separator: ", "))"
    }

    func cancelMarkRoom() {
        pendingRoomPosition = nil
    }

    // MARK: - Alignment (подгонка привязки к скану по лидару)

    /// После привязки по фотодеталям скан может лечь со сдвигом. Телефон несколько секунд
    /// собирает живую геометрию лидара, подгоняет её к сохранённому скану и сдвигает
    /// систему координат сессии на найденную поправку. После этого и карта в AR,
    /// и позы записываемых кадров — в координатах скана.
    func alignToScan() {
        guard mode == .relocalized, let mesh = relocalizedMesh, !isRecording else { return }
        cancelAlignment()
        isAligning = true
        status = "Aligning: slowly sweep the phone over walls and furniture…"
        let targetPoints = mesh.positions
        let targetNormals = MeshIO.vertexNormals(mesh)
        alignTask = Task { [weak self] in
            // Ждём, пока лидар соберёт достаточно геометрии (минимум 4 с, максимум 20 с).
            var live: [SIMD3<Float>] = []
            for second in 1...20 {
                try? await Task.sleep(for: .seconds(1))
                guard let self, !Task.isCancelled else { return }
                if let frame = self.arView.session.currentFrame {
                    live = MeshIO.collect(from: frame.anchors).positions
                }
                if second >= 4 && live.count >= 4000 { break }
            }
            guard let self, !Task.isCancelled else { return }
            let source = Aligner.subsample(live, maxCount: 2000)
            let result = await Task.detached(priority: .userInitiated) {
                Aligner.align(source: source, targetPoints: targetPoints, targetNormals: targetNormals)
            }.value
            guard !Task.isCancelled else { return }
            self.isAligning = false
            guard self.mode == .relocalized else { return }
            self.finishAlignment(result, livePoints: live.count)
        }
    }

    private func finishAlignment(_ result: AlignmentResult?, livePoints: Int) {
        guard let result else {
            status = "Not enough LiDAR data yet (\(livePoints) points). Look around the room, then tap Align."
            return
        }
        let shift = String(format: "%.1f cm", result.shift * 100)
        let turn = String(format: "%.1f°", result.yawDegrees)
        let fit = String(format: "±%.1f cm", result.fitRMS * 100)
        let matched = Int((result.matchedFraction * 100).rounded())
        guard result.isReliable else {
            status = "Alignment uncertain (fit \(fit), \(matched)% matched). Not applied. Look around more, then tap Align."
            return
        }
        // Новый мировой ноль, заданный в текущих координатах: обратная поправка.
        arView.session.setWorldOrigin(relativeTransform: simd_inverse(result.transform))
        status = "Aligned: corrected \(shift), \(turn). Fit \(fit), \(matched)% matched. Ready to Record."
    }

    private func cancelAlignment() {
        alignTask?.cancel()
        alignTask = nil
        isAligning = false
    }

    // MARK: - Sharing

    func shareSelected() {
        guard let sessionDir = selectedSession?.url else {
            status = "No saved session selected"
            return
        }
        do {
            shareItem = ShareItem(url: try SessionFiles.zip(sessionDir))
        } catch {
            status = "Zip failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Recording (камера на штативе)

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        guard mode == .relocalized, let sessionDir = relocalizedSessionDir else {
            status = "Relocalize first, then Record"
            return
        }
        do {
            let dir = try SessionFiles.makeFramesDir(sessionDir: sessionDir)
            framesDir = dir
            nextFrameIndex = SessionFiles.nextFrameIndex(framesDir: dir)
            currentTake = SessionFiles.nextTake(framesDir: dir)
        } catch {
            status = "Cannot create capture folder: \(error.localizedDescription)"
            return
        }
        recordedFrames = 0
        isRecording = true
        UIApplication.shared.isIdleTimerDisabled = true   // экран не гаснет, пока телефон на штативе
        status = "Recording take \(currentTake), 1 frame per second… Tap Stop when done."
        recordTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                self?.captureFrame()
            }
        }
    }

    private func stopRecording() {
        recordTask?.cancel()
        recordTask = nil
        isRecording = false
        UIApplication.shared.isIdleTimerDisabled = false
        refreshSessions()
        status = "Take \(currentTake): \(recordedFrames) frames. Next room: carry the phone there (app open), Align, Record. Or Share."
    }

    private func stopRecordingIfNeeded() {
        if isRecording { stopRecording() }
    }

    /// Раз в секунду: кадр высокого разрешения (если поддерживается), иначе обычный кадр видео.
    private func captureFrame() {
        guard isRecording, let framesDir else { return }
        guard let current = arView.session.currentFrame, case .normal = current.camera.trackingState else {
            status = "Tracking limited — frame skipped"
            return
        }
        let index = nextFrameIndex
        if highResSupported {
            guard !highResInFlight else { return }          // предыдущий снимок ещё не готов — пропускаем
            highResInFlight = true
            nextFrameIndex += 1
            let maxWidth = maxFrameWidth
            arView.session.captureHighResolutionFrame { [weak self] frame, error in
                // Приходит асинхронно, не обязательно в главном потоке: собираем JPEG здесь,
                // а в главный поток передаём только готовые данные.
                let packet = frame.flatMap { FramePacket.make(from: $0, maxWidth: maxWidth) }
                let failure = packet == nil ? "High-res capture failed: \(error?.localizedDescription ?? "encoding error")" : nil
                let target = self
                Task { @MainActor in
                    target?.highResInFlight = false
                    if let packet {
                        target?.writeFrame(packet, index: index, to: framesDir)
                    } else if let failure {
                        target?.status = failure
                    }
                }
            }
        } else {
            nextFrameIndex += 1
            guard let packet = FramePacket.make(from: current, maxWidth: maxFrameWidth) else {
                status = "JPEG encoding failed"
                return
            }
            writeFrame(packet, index: index, to: framesDir)
        }
    }

    /// Запись кадра и его JSON (docs/data_spec.md, раздел 4.3).
    private func writeFrame(_ packet: FramePacket, index: Int, to framesDir: URL) {
        let name = String(format: "%06d", index)
        let meta = FrameMeta(
            index: index,
            timestamp: packet.timestamp,
            wall_time_utc: SessionFiles.isoNow(),
            image: .init(file: "frames/\(name).jpg", width: packet.width, height: packet.height),
            camera: .init(
                T_world_from_camera_ar: packet.transformRowMajor,
                K: packet.intrinsicsRowMajor,
                tracking_state: packet.trackingNormal ? "normal" : "limited"),
            take: currentTake)
        do {
            try packet.jpeg.write(to: framesDir.appendingPathComponent("\(name).jpg"), options: .atomic)
            try SessionFiles.writeJSON(meta, to: framesDir.appendingPathComponent("\(name).json"))
            recordedFrames += 1
            status = "Recording… \(recordedFrames) frames (\(packet.width)×\(packet.height))"
        } catch {
            status = "Frame save failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Frame updates

    fileprivate func apply(time: TimeInterval, tracking: String, isNormal: Bool, mapping: String, meshAnchors: Int) {
        if mode == .relocalizing && !isNormal {
            if let start = relocalizeStartTime {
                if time - start > 30 && !relocalizeHintShown {
                    relocalizeHintShown = true
                    status = "Still searching. Stand where the scan started, use the same lighting, aim at furniture — or make a new scan."
                }
            } else {
                relocalizeStartTime = time
            }
        }
        if mode == .relocalizing && isNormal {
            overlay?.isEnabled = true
            mode = .relocalized
            alignToScan()
        }
        guard time - lastUIUpdate > 0.3 else { return }
        lastUIUpdate = time
        self.tracking = tracking
        self.mapping = mapping
        liveChunks = (mode == .scanning || mode == .saving) ? "Mesh chunks: \(meshAnchors)" : ""
    }

    nonisolated static func describe(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "normal"
        case .notAvailable:
            return "not available"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited: initializing"
            case .relocalizing: return "limited: relocalizing"
            case .excessiveMotion: return "limited: moving too fast"
            case .insufficientFeatures: return "limited: not enough detail"
            @unknown default: return "limited"
            }
        @unknown default:
            return "unknown"
        }
    }

    nonisolated static func describe(_ status: ARFrame.WorldMappingStatus) -> String {
        switch status {
        case .notAvailable: return "not available"
        case .limited: return "limited"
        case .extending: return "extending"
        case .mapped: return "mapped"
        @unknown default: return "unknown"
        }
    }
}

extension ScanController: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let time = frame.timestamp
        let trackingText = Self.describe(frame.camera.trackingState)
        var isNormal = false
        if case .normal = frame.camera.trackingState { isNormal = true }
        let mappingText = Self.describe(frame.worldMappingStatus)
        let meshAnchors = frame.anchors.filter { $0 is ARMeshAnchor }.count
        let normal = isNormal
        // ARKit вызывает делегата в главном потоке (очередь по умолчанию).
        MainActor.assumeIsolated {
            self.apply(time: time, tracking: trackingText, isNormal: normal, mapping: mappingText, meshAnchors: meshAnchors)
        }
    }
}
