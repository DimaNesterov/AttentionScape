import Foundation
import simd

/// Отметка комнаты: название и точка, где стоял телефон, в координатах скана (W).
struct RoomMark: Codable, Identifiable, Hashable {
    var name: String
    var position: [Float]          // x, y, z (м)
    var created_utc: String

    var id: String { name }
    var point: SIMD3<Float> {
        position.count == 3 ? SIMD3(position[0], position[1], position[2]) : .zero
    }
}

/// scan/rooms.json — отметки комнат. Отдельный файл: скан и его mesh_sha256 не меняются.
struct RoomsFile: Codable {
    var schema = "attentionscape/rooms/0.1"
    var rooms: [RoomMark] = []
}

/// Доли внимания по комнатам из results/attention.json (поле "rooms", пишет Mac).
struct RoomShare: Decodable {
    let name: String
    let share: Float
}

enum Rooms {

    static func url(sessionDir: URL) -> URL {
        sessionDir
            .appendingPathComponent("scan", isDirectory: true)
            .appendingPathComponent("rooms.json")
    }

    static func load(sessionDir: URL) -> [RoomMark] {
        guard let data = try? Data(contentsOf: url(sessionDir: sessionDir)),
              let file = try? JSONDecoder().decode(RoomsFile.self, from: data) else { return [] }
        return file.rooms
    }

    static func save(_ rooms: [RoomMark], sessionDir: URL) throws {
        try SessionFiles.writeJSON(RoomsFile(rooms: rooms), to: url(sessionDir: sessionDir))
    }

    /// Новая отметка с тем же названием заменяет старую (так исправляется неудачная отметка).
    static func upsert(_ mark: RoomMark, into rooms: [RoomMark]) -> [RoomMark] {
        rooms.filter { $0.name.caseInsensitiveCompare(mark.name) != .orderedSame } + [mark]
    }

    static func make(name: String, position: SIMD3<Float>) -> RoomMark {
        RoomMark(name: name, position: [position.x, position.y, position.z], created_utc: SessionFiles.isoNow())
    }

    /// Доли внимания по комнатам, если карта уже посчитана с отметками комнат.
    static func shares(sessionDir: URL) -> [String: Float] {
        struct Header: Decodable { let rooms: [RoomShare]? }
        let jsonURL = sessionDir
            .appendingPathComponent("results", isDirectory: true)
            .appendingPathComponent("attention.json")
        guard let data = try? Data(contentsOf: jsonURL),
              let header = try? JSONDecoder().decode(Header.self, from: data) else { return [:] }
        return Dictionary((header.rooms ?? []).map { ($0.name, $0.share) }, uniquingKeysWith: { a, _ in a })
    }
}
