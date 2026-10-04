import Foundation
import RealityKit
import SwiftUI
import UIKit

struct ScanView: View {
    @StateObject private var controller = ScanController()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARViewContainer(arView: controller.arView)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                Text(controller.mode.rawValue)
                    .font(.headline)
                Text(controller.status)
                    .font(.subheadline)
                Text("Tracking: \(controller.tracking) · Map: \(controller.mapping)")
                    .font(.caption)
                if !controller.liveChunks.isEmpty {
                    Text(controller.liveChunks).font(.caption)
                }
                if !controller.meshInfo.isEmpty {
                    Text(controller.meshInfo).font(.caption)
                }

                HStack {
                    Button("Start scan") { controller.startScan() }
                        .disabled(controller.mode == .saving || controller.isRecording)
                    Button("Save") { controller.saveScan() }
                        .disabled(controller.mode != .scanning)
                }
                HStack {
                    Button("Relocalize") { controller.relocalizeLatest() }
                        .disabled(controller.mode == .saving || controller.isRecording)
                    Button("Share") { controller.shareLatest() }
                        .disabled(controller.mode == .saving || controller.isRecording)
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
        }
        .sheet(item: $controller.shareItem) { item in
            ActivityView(items: [item.url])
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
