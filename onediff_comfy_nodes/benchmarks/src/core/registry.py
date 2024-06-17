from typing import Callable, Dict


def create_constructor_registry():
    # Dictionary to hold registered constructors
    constructor_registry: Dict[str, Callable] = {}

    def register(workflow_api_file_path) -> Callable:
        def decorator(construct_input_function: Callable) -> Callable:
            constructor_registry[workflow_api_file_path] = construct_input_function
            return construct_input_function

        return decorator

    def dispatch(workflow_api_file_path: str, *args, **kwargs):
        constructor = constructor_registry.get(workflow_api_file_path)
        if constructor is None:
            raise ValueError(f"No constructor registered for {workflow_api_file_path}")
        return constructor(workflow_api_file_path, *args, **kwargs)

    return register, dispatch
