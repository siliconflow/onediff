import itertools
from typing import Callable, Dict, List, NamedTuple, Tuple, Union


def create_generator_registry():
    # Dictionary to hold registered constructors
    generator_registry: Dict[str, Callable] = {}

    def register(workflow_path: Union[List[str], str]) -> Callable:
        def decorator(generator_function: Callable) -> Callable:
            if isinstance(workflow_path, (List, Tuple)):
                for workflow in workflow_path:
                    generator_registry[workflow] = generator_function
            else:
                generator_registry[workflow_path] = generator_function
            return generator_function

        return decorator

    def dispatch(workflow_path: Union[List[str], str], *args, **kwargs) -> NamedTuple:
        if isinstance(workflow_path, (List, Tuple)):
            return itertools.chain(
                *[dispatch(w, *args, **kwargs) for w in workflow_path]
            )
        else:
            generator = generator_registry.get(workflow_path)
            if generator is None:
                raise ValueError(f"No generator registered for {workflow_path}")
            return generator(workflow_path, *args, **kwargs)

    return register, dispatch
