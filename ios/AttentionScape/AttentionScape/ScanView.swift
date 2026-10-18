import Foundation
import RealityKit
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ScanView: View {
    @StateObject private var controller = ScanController()
    @State private var showSessions = false
    @State private var showImporter = false
    @State private var dollhouseSession: SessionSummary?
    @AppStorage("panelCollapsed") private var collapsed = false
    @State private var showRoomPrompt = false
    @State private var roomName = ""

    var body: some View {
        ZStack(alignment: .bottom) {
            ARViewContainer(arView: controller.arView)
                .ignoresSafeArea()
            if collapsed {
                collapsedBar
            } else {
                panel
            }
        }
        .sheet(isPresented: $showSessions) {
            SessionListView(controller: controller)
        }
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.data, .folder],
                      allowsMultipleSelection: true) { result in
            controller.importAttention(result)
        }
        .fullScreenCover(item: $dollhouseSession) { session in
            DollhouseView(session: session)
        }
        .alert("Room name", isPresented: $showRoomPrompt) {
            TextField("e.g. Kitchen", text: $roomName)
            Button("Save") { controller.saveRoom(name: roomName) }
            Button("Cancel", role: .cancel) { controller.cancelMarkRoom() }
        } message: {
            Text("Marks this spot as the room. Same name again replaces the old mark.")
        }
    }

    /// Свёрнутое меню: одна строка с режимом (и записью), кнопка разворачивает панель.
    private var collapsedBar: some View {
        HStack(spacing: 8) {
            Text(controller.mode.rawValue)
                .font(.subheadline.weight(.semibold))
            if controller.isRecording {
                Text("● \(controller.recordedFrames) frames")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            Spacer()
            Button {
                withAnimation(.snappy) { collapsed = false }
            } label: {
                Image(systemName: "chevron.up")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
        .padding()
    }

    private var panel: some View {
        let busy = controller.mode == .saving || controller.isRecording
        let hasMap = controller.selectedSession?.hasMap ?? false
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(controller.mode.rawValue)
                    .font(.headline)
                Spacer()
                Button {
                    withAnimation(.snappy) { collapsed = true }
                } label: {
                    Image(systemName: "chevron.down")
                }
                .buttonStyle(.bordered)
            }
            Text(controller.status)
                .font(.subheadline)
            Text("Tracking: \(controller.tracking) · Map: \(controller.mapping)")
                .font(.caption)
            if let session = controller.selectedSession {
                Text("Session: \(session.id) · \(session.frames) frames\(session.hasMap ? " · attention map" : "")")
                    .font(.caption)
            } else {
                Text("Session: none")
                    .font(.caption)
            }
            if !controller.liveChunks.isEmpty {
                Text(controller.liveChunks).font(.caption)
            }
            if !controller.meshInfo.isEmpty {
                Text(controller.meshInfo).font(.caption)
            }
            if !controller.rooms.isEmpty {
                Text("Rooms: \(controller.rooms.map(\.name).joined(separator: ", "))")
                    .font(.caption)
            }
            if controller.showingAttention {
                AttentionLegend()
                HStack {
                    Text("Opacity").font(.caption)
                    Slider(value: $controller.heatOpacity, in: 0.1...1)
                }
            }

            HStack {
                Button("Start scan") { controller.startScan() }
                    .disabled(busy)
                Button("Save") { controller.saveScan() }
                    .disabled(controller.mode != .scanning)
                Button("Mark room") {
                    if controller.beginMarkRoom() {
                        roomName = ""
                        showRoomPrompt = true
                    }
                }
                .disabled(!controller.canMarkRoom)
            }
            HStack {
                Button("Sessions") { showSessions = true }
                    .disabled(busy)
                Button("Relocalize") { controller.relocalizeSelected() }
                    .disabled(busy || controller.selectedSession == nil)
                Button("Share") { controller.shareSelected() }
                    .disabled(busy || controller.selectedSession == nil)
            }
            HStack {
                Button(controller.isAligning ? "Aligning…" : "Align") { controller.alignToScan() }
                    .disabled(controller.mode != .relocalized || controller.isAligning || controller.isRecording)
                Button(controller.isRecording ? "Stop" : "Record") { controller.toggleRecording() }
                    .disabled(controller.mode != .relocalized || controller.isAligning)
                    .tint(controller.isRecording ? .red : nil)
                if controller.isRecording {
                    Text("● \(controller.recordedFrames) frames")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            HStack {
                Button("Import map") { showImporter = true }
                    .disabled(busy)
                Button(controller.overlayKind == .attention ? "Show scan" : "Show map") {
                    controller.toggleOverlayKind()
                }
                .disabled(busy || !hasMap)
                Button("3D view") { dollhouseSession = controller.selectedSession }
                    .disabled(busy || controller.selectedSession == nil)
            }
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding()
        .sheet(item: $controller.shareItem) { item in
            ActivityView(items: [item.url])
        }
    }
}

/// Шкала цветов карты внимания (та же, что в AR и на Mac).
struct AttentionLegend: View {
    var body: some View {
        let stops = zip(AttentionMap.stops, AttentionMap.colors).map { stop, rgb in
            Gradient.Stop(color: Color(red: Double(rgb.x) / 255, green: Double(rgb.y) / 255, blue: Double(rgb.z) / 255),
                          location: CGFloat(stop))
        }
        return HStack(spacing: 6) {
            Text("low").font(.caption2)
            LinearGradient(stops: stops, startPoint: .leading, endPoint: .trailing)
                .frame(height: 8)
                .clipShape(Capsule())
            Text("high attention").font(.caption2)
        }
    }
}

/// Список сохранённых сессий: тап — выбрать, свайп влево — удалить.
struct SessionListView: View {
    @ObservedObject var controller: ScanController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(controller.sessions) { session in
                    Button {
                        controller.selectSession(session.id)
                        dismiss()
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(session.id)
                                    .font(.body.monospaced())
                                Text("\(session.vertices) vertices · \(session.frames) frames\(session.hasMap ? " · attention map" : "")")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if session.id == controller.selectedSessionID {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                    .foregroundStyle(.primary)
                }
                .onDelete { offsets in
                    let ids = offsets.map { controller.sessions[$0].id }
                    for id in ids {
                        controller.deleteSession(id)
                    }
                }
            }
            .overlay {
                if controller.sessions.isEmpty {
                    ContentUnavailableView("No saved sessions", systemImage: "cube.transparent")
                }
            }
            .navigationTitle("Sessions")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    let arView: ARView
    func makeUIView(context: Context) -> ARView { arView }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

struct ActivityView: UIViewControllerRepresentable {
    let items: [URL]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
