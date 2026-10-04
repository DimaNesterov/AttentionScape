import Darwin
import Foundation
import simd

/// Содержимое sessions/<id>/session.json (см. docs/data_spec.md, раздел 4.1).
struct SessionInfo: Codable {
    var schema = "attentionscape/0.1"
    var session_id: String
    var created_utc: String
    var device: String
    var ios_version: String
    var app_version: String
    var mesh_sha256: String
    var n_vertices: Int
    var n_faces: Int
    var notes = ""
}

/// Краткое описание сессии для списка в приложении.
struct SessionSummary: Identifiable, Hashable {
    let id: String      // имя папки = session_id
    let url: URL
    let vertices: Int
    let faces: Int
    let frames: Int
}

private struct SessionHeader: Decodable {
    let n_vertices: Int?
    let n_faces: Int?
}

/// Метаданные кадра: capture/frames/NNNNNN.json (docs/data_spec.md, раздел 4.3).
struct FrameMeta: Codable {
    struct ImageInfo: Codable {
        var file: String
        var width: Int
        var height: Int
    }
    struct CameraInfo: Codable {
        var T_world_from_camera_ar: [Float]   // 4×4 построчно
        var K: [Float]                        // 3×3 построчно
        var tracking_state: String
    }
    var index: Int
    var timestamp: Double
    var wall_time_utc: String
    var image: ImageInfo
    var camera: CameraInfo
}

enum SessionFiles {

    /// Documents/sessions — корень всех сессий на iPhone.
    static var sessionsRoot: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("sessions", isDirectory: true)
    }

    /// Формат: YYYYMMDD-HHMMSS-<name>
    static func newSessionID(name: String = "scan") -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: Date()))-\(name)"
    }

    /// Создаёт sessions/<id>/scan/ и возвращает путь к ней.
    static func makeScanDir(sessionID: String) throws -> URL {
        let url = sessionsRoot
            .appendingPathComponent(sessionID, isDirectory: true)
            .appendingPathComponent("scan", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Самая свежая ПОЛНАЯ сессия: есть session.json, scan/mesh.ply и scan/worldmap.arworldmap.
    /// Неполные папки (например, если карта мира не сохранилась) пропускаются.
    static func latestSessionDir() -> URL? {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: sessionsRoot, includingPropertiesForKeys: [.isDirectoryKey]) else { return nil }
        return items
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .filter { isComplete($0) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .last
    }

    static func isComplete(_ sessionDir: URL) -> Bool {
        let fm = FileManager.default
        let scan = sessionDir.appendingPathComponent("scan", isDirectory: true)
        return fm.fileExists(atPath: sessionDir.appendingPathComponent("session.json").path)
            && fm.fileExists(atPath: scan.appendingPathComponent("mesh.ply").path)
            && fm.fileExists(atPath: scan.appendingPathComponent("worldmap.arworldmap").path)
    }

    /// Все полные сессии, новые сверху.
    static func listCompleteSessions() -> [SessionSummary] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: sessionsRoot, includingPropertiesForKeys: [.isDirectoryKey]) else { return [] }
        return items
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .filter { isComplete($0) }
            .map { dir in
                let header = (try? Data(contentsOf: dir.appendingPathComponent("session.json")))
                    .flatMap { try? JSONDecoder().decode(SessionHeader.self, from: $0) }
                return SessionSummary(
                    id: dir.lastPathComponent,
                    url: dir,
                    vertices: header?.n_vertices ?? 0,
                    faces: header?.n_faces ?? 0,
                    frames: frameCount(sessionDir: dir))
            }
            .sorted { $0.id > $1.id }
    }

    static func frameCount(sessionDir: URL) -> Int {
        let dir = sessionDir
            .appendingPathComponent("capture", isDirectory: true)
            .appendingPathComponent("frames", isDirectory: true)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return names.filter { $0.hasSuffix(".json") }.count
    }

    /// Удаляет старые архивы для отправки (вызывается при запуске приложения).
    static func cleanupTemporaryZips() {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory
        let names = (try? fm.contentsOfDirectory(atPath: tmp.path)) ?? []
        for name in names where name.hasSuffix(".zip") {
            try? fm.removeItem(at: tmp.appendingPathComponent(name))
        }
    }

    static func isoNow() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    static func appVersion() -> String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0"
    }

    /// Идентификатор модели, например "iPhone18,1".
    static func machineIdentifier() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    /// Упаковывает папку сессии в .zip (для отправки на Mac).
    /// У каждого архива своё имя, поэтому повторное нажатие Share не ломает идущую передачу.
    static func zip(_ dir: URL) throws -> URL {
        let stamp = Int(Date().timeIntervalSince1970)
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(dir.lastPathComponent)-\(stamp).zip")
        var coordinatorError: NSError?
        var copyError: Error?
        var copied = false
        NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error: &coordinatorError) { zipURL in
            do {
                try FileManager.default.copyItem(at: zipURL, to: destination)
                copied = true
            } catch {
                copyError = error
            }
        }
        if let coordinatorError { throw coordinatorError }
        if let copyError { throw copyError }
        guard copied else { throw CocoaError(.fileWriteUnknown) }
        return destination
    }

    // MARK: - Capture

    /// Создаёт sessions/<id>/capture/frames/ и возвращает путь к ней.
    static func makeFramesDir(sessionDir: URL) throws -> URL {
        let url = sessionDir
            .appendingPathComponent("capture", isDirectory: true)
            .appendingPathComponent("frames", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Следующий свободный номер кадра (повторная запись продолжает нумерацию).
    static func nextFrameIndex(framesDir: URL) -> Int {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: framesDir.path)) ?? []
        let used = names.compactMap { name -> Int? in
            guard name.hasSuffix(".json") else { return nil }
            return Int(name.dropLast(5))
        }
        return (used.max() ?? 0) + 1
    }

    /// simd хранит матрицы по столбцам; в JSON пишем построчно (data_spec.md, раздел 2).
    static func rowMajor(_ m: simd_float4x4) -> [Float] {
        (0..<4).flatMap { r in [m.columns.0[r], m.columns.1[r], m.columns.2[r], m.columns.3[r]] }
    }

    static func rowMajor(_ m: simd_float3x3) -> [Float] {
        (0..<3).flatMap { r in [m.columns.0[r], m.columns.1[r], m.columns.2[r]] }
    }
}
