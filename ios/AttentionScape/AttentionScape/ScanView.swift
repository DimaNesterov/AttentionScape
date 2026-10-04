import Foundation
import RealityKit
import SwiftUI
import UIKit

struct ScanView: View {
    @StateObject private var controller = ScanController()
    @State private var showSessions = false

    var body: some View {
        ZStack(alignment: .bottom) {
            ARViewContainer(arView: controller.arView)
                .ignoresSafeArea()
            panel
        }
        .sheet(isPresented: $showSessions) {
            SessionListView(controller: controller)
        }
    }

    private var panel: some View {
        let busy = controller.mode == .saving || controller.isRecording
        return VStack(alignment: .leading, spacing: 6) {
            Text(controller.mode.rawValue)
                .font(.headline)
            Text(controller.status)
                .font(.subheadline)
            Text("Tracking: \(controller.tracking) · Map: \(controller.mapping)")
                .font(.caption)
            if let session = controller.selectedSession {
                Text("Session: \(session.id) · \(session.frames) frames")
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

            HStack {
                Button("Start scan") { controller.startScan() }
                    .disabled(busy)
                Button("Save") { controller.saveScan() }
                    .disabled(controller.mode != .scanning)
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
                Button(controller.isRecording ? "Stop" : "Record") { controller.toggleRecording() }
                    .disabled(controller.mode != .relocalized)
                    .tint(controller.isRecording ? .red : nil)
                if controller.isRecording {
                    Text("● \(controller.recordedFrames) frames")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
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
                                Text("\(session.vertices) vertices · \(session.frames) frames")
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
