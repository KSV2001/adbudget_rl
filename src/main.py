from src.gradio_app import build_interface

if __name__ == "__main__":
    demo = build_interface()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "8090")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share = True,
    )