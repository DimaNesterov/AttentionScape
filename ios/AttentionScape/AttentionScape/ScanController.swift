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

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)

    private var overlay: AnchorEntity?
    private var lastUIUpdate: TimeInterval = 0

    // Запись кадров (неделя 2)
    private var relocalizedSessionDir: URL?
    private var framesDir: URL?
    private var nextFrameIndex = 1
    private var recordTask: Task<Void, Never>?
    private let ciContext = CIContext()

    override init() {
        super.init()
        arView.session.delegate = self
    }

    // MARK: - Scanning

    func startScan() {
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) else {
            status = "This device does not support LiDAR scene reconstruction"
            return
        }
        stopRecordingIfNeeded()
        removeOverlay()
        relocalizedSessionDir = nil
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        config.environmentTexturing = .none
        arView.debugOptions = [.showSceneUnderstanding]
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        mode = .scanning
        meshInfo = ""
        status = "Walk slowly around the room. Save when Map shows \"mapped\"."
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
        // Скан завершён: останавливаем сканирование, новый — только по кнопке Start scan.
        arView.session.pause()
        arView.debugOptions = []
        liveChunks = ""
        meshInfo = "\(info.n_vertices) vertices, \(info.n_faces) triangles"
        mode = .idle
        status = "Saved \(info.session_id). Tap Relocalize to check it, or Start scan for a new one."
    }

    // MARK: - Relocalization

    func relocalizeLatest() {
        guard let sessionDir = SessionFiles.latestSessionDir() else {
            status = "No complete saved session yet"
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

            let model = try MeshIO.classColoredModel(mesh, opacity: 0.7)
            let anchor = AnchorEntity(world: .zero)
            anchor.addChild(model)
            anchor.isEnabled = false

            removeOverlay()
            arView.scene.addAnchor(anchor)
            overlay = anchor

            let config = ARWorldTrackingConfiguration()
            config.initialWorldMap = worldMap
            arView.debugOptions = []
            arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])

            relocalizedSessionDir = sessionDir
            mode = .relocalizing
            liveChunks = ""
            meshInfo = "\(sessionDir.lastPathComponent): \(mesh.positions.count) vertices"
            status = "Point the phone at the scanned area, ideally from where you started."
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
    }

    private func removeOverlay() {
        if let overlay {
            arView.scene.removeAnchor(overlay)
            self.overlay = nil
        }
    }

    // MARK: - Sharing

    func shareLatest() {
        guard let sessionDir = SessionFiles.latestSessionDir() else {
            status = "No complete saved session yet"
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
        } catch {
            status = "Cannot create capture folder: \(error.localizedDescription)"
            return
        }
        recordedFrames = 0
        isRecording = true
        UIApplication.shared.isIdleTimerDisabled = true   // экран не гаснет, пока телефон на штативе
        status = "Recording 1 frame per second… Tap Stop when done."
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
        status = "Recorded \(recordedFrames) frames. Tap Share to send the session to the Mac."
    }

    private func stopRecordingIfNeeded() {
        if isRecording { stopRecording() }
    }

    /// Сохраняет текущий кадр камеры и его метаданные (data_spec.md, раздел 4.3).
    private func captureFrame() {
        guard isRecording, let framesDir, let frame = arView.session.currentFrame else { return }
        guard case .normal = frame.camera.trackingState else {
            status = "Tracking limited — frame skipped"
            return
        }
        let buffer = frame.capturedImage                  // ландшафтная ориентация сенсора, без поворота
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let image = CIImage(cvPixelBuffer: buffer)
        guard let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]) else {
            status = "JPEG encoding failed"
            return
        }
        let name = String(format: "%06d", nextFrameIndex)
        let meta = FrameMeta(
            index: nextFrameIndex,
            timestamp: frame.timestamp,
            wall_time_utc: SessionFiles.isoNow(),
            image: .init(file: "frames/\(name).jpg", width: width, height: height),
            camera: .init(
                T_world_from_camera_ar: SessionFiles.rowMajor(frame.camera.transform),
                K: SessionFiles.rowMajor(frame.camera.intrinsics),
                tracking_state: "normal"))
        do {
            try jpeg.write(to: framesDir.appendingPathComponent("\(name).jpg"), options: .atomic)
            try SessionFiles.writeJSON(meta, to: framesDir.appendingPathComponent("\(name).json"))
            nextFrameIndex += 1
            recordedFrames += 1
            status = "Recording… \(recordedFrames) frames"
        } catch {
            status = "Frame save failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Frame updates

    fileprivate func apply(time: TimeInterval, tracking: String, isNormal: Bool, mapping: String, meshAnchors: Int) {
        if mode == .relocalizing && isNormal {
            overlay?.isEnabled = true
            mode = .relocalized
            status = "Relocalized. Put the phone on the tripod and tap Record."
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
