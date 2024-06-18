from typing import Callable, Dict, List, NamedTuple, Union, Tuple


def create_constructor_registry():
    # Dictionary to hold registered constructors
    constructor_registry: Dict[str, Callable] = {}

    def register(workflow_api_file_path: Union[List[str], str]) -> Callable:
        def decorator(construct_input_function: Callable) -> Callable:
            if isinstance(workflow_api_file_path, list):
                for workflow in workflow_api_file_path:
                    constructor_registry[workflow] = construct_input_function
            else:
                constructor_registry[workflow_api_file_path] = construct_input_function
            return construct_input_function

        return decorator

    def dispatch(workflow_api_file_path: str, *args, **kwargs) -> NamedTuple:
        constructor = constructor_registry.get(workflow_api_file_path)
        if constructor is None:
            raise ValueError(f"No constructor registered for {workflow_api_file_path}")
        return constructor(workflow_api_file_path, *args, **kwargs)

    return register, dispatch
