from pathlib import Path

hints_message = """
                    <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #31708f;">
                            Hints Message
                        </div>
                        <div style="padding: 10px; border: 1px solid #31708f; border-radius: 5px; background-color: #f9f9f9;">
                            Hints: Enterprise function is not supported on your system.
                        </div>
                        <p style="margin-top: 15px;">
                            If you need Enterprise Level Support for your system or business, please send an email to 
                            <a href="mailto:business@siliconflow.com" style="color: #31708f; text-decoration: none;">business@siliconflow.com</a>.
                            <br>
                            Tell us about your use case, deployment scale, and requirements.
                        </p>
                        <p>
                            <strong>GitHub Issue:</strong>
                            <a href="https://github.com/siliconflow/onediff/issues" style="color: #31708f; text-decoration: none;">https://github.com/siliconflow/onediff/issues</a>
                        </p>
                    </div>
                    """

graph_checkpoints = []

def graph_checkpoints_path():
    import modules.shared as shared
    return shared.opts.onediff_graph_save_path

def get_graph_checkpoints():
    global graph_checkpoints
    if len(graph_checkpoints) == 0:
        refresh_graph_checkpoints()
    return graph_checkpoints

def refresh_graph_checkpoints(path: Path = None):
    global graph_checkpoints
    path = path or graph_checkpoints_path()
    graph_checkpoints = [f.stem for f in Path(path).iterdir() if f.is_file()]  