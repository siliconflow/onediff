import importlib
import importlib.metadata
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from huggingface_hub import (
    ModelCard,
    create_repo,
    hf_hub_download,
    model_info,
    snapshot_download,
)
from packaging import version
from tqdm.auto import tqdm

import diffusers

from diffusers import __version__
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BaseOutput,
    deprecate,
    get_class_from_dynamic_module,
    is_accelerate_available,
    is_accelerate_version,
    is_torch_version,
    is_transformers_available,
    logging,
    numpy_to_pil,
)

diffusers_version = version.parse(importlib.metadata.version("diffusers"))
if diffusers_version < version.parse("0.25.0"):
    from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE

    token_arg_name = "use_auth_token"
else:
    DIFFUSERS_CACHE = None
    HF_HUB_OFFLINE = None
    token_arg_name = "token"

from diffusers.utils.torch_utils import is_compiled_module

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, maybe_raise_or_warn


if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel
    from transformers.utils import FLAX_WEIGHTS_NAME as TRANSFORMERS_FLAX_WEIGHTS_NAME
    from transformers.utils import SAFE_WEIGHTS_NAME as TRANSFORMERS_SAFE_WEIGHTS_NAME
    from transformers.utils import WEIGHTS_NAME as TRANSFORMERS_WEIGHTS_NAME

from diffusers.utils import (
    FLAX_WEIGHTS_NAME,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    PushToHubMixin,
)


if is_accelerate_available():
    import accelerate


INDEX_FILE = "diffusion_pytorch_model.bin"
CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "diffusers.utils"
TRANSFORMERS_DUMMY_MODULES_FOLDER = "transformers.utils"
CONNECTED_PIPES_KEYS = ["prior"]


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {"ORTModule": ["save_pretrained", "from_pretrained"],},
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def get_class_obj_and_candidates(
    library_name,
    class_name,
    importable_classes,
    pipelines,
    is_pipeline_module,
    component_name=None,
    cache_dir=None,
):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    component_folder = os.path.join(cache_dir, component_name)

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)

        class_obj = getattr(pipeline_module, class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    elif os.path.isfile(os.path.join(component_folder, library_name + ".py")):
        # load custom component
        class_obj = get_class_from_dynamic_module(
            component_folder, module_file=library_name + ".py", class_name=class_name
        )
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        # else we just import it from the library.
        if class_name == "UNet2DConditionModel":
            library_name = "onediffx.deep_cache.models.unet_2d_condition"

        if class_name == "UNetSpatioTemporalConditionModel":
            assert diffusers_version >= version.parse("0.24.0"), (
                "SVD not support in diffusers-" + diffusers_version
            )
            library_name = "onediffx.deep_cache.models.unet_spatio_temporal_condition"

        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {
            c: getattr(library, c, None) for c in importable_classes.keys()
        }

    return class_obj, class_candidates


def _get_pipeline_class(
    class_obj,
    config,
    load_connected_pipeline=False,
    custom_pipeline=None,
    cache_dir=None,
    revision=None,
):
    if custom_pipeline is not None:
        if custom_pipeline.endswith(".py"):
            path = Path(custom_pipeline)
            # decompose into folder & file
            file_name = path.name
            custom_pipeline = path.parent.absolute()
        else:
            file_name = CUSTOM_PIPELINE_FILE_NAME

        return get_class_from_dynamic_module(
            custom_pipeline,
            module_file=file_name,
            cache_dir=cache_dir,
            revision=revision,
        )

    if class_obj != DiffusionPipeline:
        return class_obj

    diffusers_module = importlib.import_module(class_obj.__module__.split(".")[0])
    class_name = config["_class_name"]

    if class_name.startswith("Flax"):
        class_name = class_name[4:]

    pipeline_cls = getattr(diffusers_module, class_name)

    if load_connected_pipeline:
        from .auto_pipeline import _get_connected_pipeline

        connected_pipeline_cls = _get_connected_pipeline(pipeline_cls)
        if connected_pipeline_cls is not None:
            logger.info(
                f"Loading connected pipeline {connected_pipeline_cls.__name__} instead of {pipeline_cls.__name__} as specified via `load_connected_pipeline=True`"
            )
        else:
            logger.info(
                f"{pipeline_cls.__name__} has no connected pipeline class. Loading {pipeline_cls.__name__}."
            )

        pipeline_cls = connected_pipeline_cls or pipeline_cls

    return pipeline_cls


def load_sub_model(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    torch_dtype: torch.dtype,
    provider: Any,
    sess_options: Any,
    device_map: Optional[Union[Dict[str, torch.device], str]],
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]],
    offload_folder: Optional[Union[str, os.PathLike]],
    offload_state_dict: bool,
    model_variants: Dict[str, str],
    name: str,
    from_flax: bool,
    variant: str,
    low_cpu_mem_usage: bool,
    cached_folder: Union[str, os.PathLike],
):
    """Helper method to load the module `name` from `library_name` and `class_name`"""
    # retrieve class candidates
    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # retrive load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(
            DUMMY_MODULES_FOLDER
        ) or none_module.startswith(TRANSFORMERS_DUMMY_MODULES_FOLDER)
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    load_method = getattr(class_obj, load_method_name)

    # add kwargs to loading method
    loading_kwargs = {}
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs["torch_dtype"] = torch_dtype
    if issubclass(class_obj, diffusers.OnnxRuntimeModel):
        loading_kwargs["provider"] = provider
        loading_kwargs["sess_options"] = sess_options

    is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)

    if is_transformers_available():
        transformers_version = version.parse(
            version.parse(transformers.__version__).base_version
        )
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )

    # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
    # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
    # This makes sure that the weights won't be initialized which significantly speeds up loading.
    if is_diffusers_model or is_transformers_model:
        loading_kwargs["device_map"] = device_map
        loading_kwargs["max_memory"] = max_memory
        loading_kwargs["offload_folder"] = offload_folder
        loading_kwargs["offload_state_dict"] = offload_state_dict
        loading_kwargs["variant"] = model_variants.pop(name, None)
        if from_flax:
            loading_kwargs["from_flax"] = True

        # the following can be deleted once the minimum required `transformers` version
        # is higher than 4.27
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            loading_kwargs.pop("variant")

        # if `from_flax` and model is transformer model, can currently not load with `low_cpu_mem_usage`
        if not (from_flax and is_transformers_model):
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            loading_kwargs["low_cpu_mem_usage"] = False

    # check if the module is in a subdirectory
    if os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(
            os.path.join(cached_folder, name), **loading_kwargs
        )
    else:
        # else load from the root directory
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    return loaded_sub_model


def from_pretrained(
    cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
):
    r"""
    Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

    The pipeline is set in evaluation mode (`model.eval()`) by default.

    If you get the error message below, you need to finetune the weights for your downstream task:

    ```
    Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
    - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    ```

    Parameters:
        pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:

                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                  hosted on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                  saved using
                [`~DiffusionPipeline.save_pretrained`].
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
            dtype is automatically derived from the model's weights.
        custom_pipeline (`str`, *optional*):

            <Tip warning={true}>

            🧪 This is an experimental feature and may change in the future.

            </Tip>

            Can be either:

                - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                  pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                  the custom pipeline.
                - A string, the *file name* of a community pipeline hosted on GitHub under
                  [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                  names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                  instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                  current main branch of GitHub.
                - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                  must contain a file called `pipeline.py` that defines the custom pipeline.

            For more information on how to load and create custom pipelines, please have a look at [Loading and
            Adding Custom
            Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
            incompletely downloaded files are deleted.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        output_loading_info(`bool`, *optional*, defaults to `False`):
            Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        custom_revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
            `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
            custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
        mirror (`str`, *optional*):
            Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
            guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
            information.
        device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn’t need to be defined for each
            parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
            same device.

            Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
            more information about each option see [designing a device
            map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
            each GPU and the available CPU RAM if unset.
        offload_folder (`str` or `os.PathLike`, *optional*):
            The path to offload weights if device_map contains the value `"disk"`.
        offload_state_dict (`bool`, *optional*):
            If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
            the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
            when there is some disk offload.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
            Speed up model loading only loading the pretrained weights and not initializing the weights. This also
            tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
            argument to `True` will raise an error.
        use_safetensors (`bool`, *optional*, defaults to `None`):
            If set to `None`, the safetensors weights are downloaded if they're available **and** if the
            safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
            weights. If set to `False`, safetensors weights are not loaded.
        use_onnx (`bool`, *optional*, defaults to `None`):
            If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
            will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
            `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
            with `.onnx` and `.pb`.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
            class). The overwritten components are passed directly to the pipelines `__init__` method. See example
            below for more information.
        variant (`str`, *optional*):
            Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
            loading `from_flax`.

    <Tip>

    To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
    `huggingface-cli login`.

    </Tip>

    Examples:

    ```py
    >>> from diffusers import DiffusionPipeline

    >>> # Download pipeline from huggingface.co and cache.
    >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

    >>> # Download pipeline that requires an authorization token
    >>> # For more information on access tokens, please refer to this section
    >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
    >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    >>> # Use a different scheduler
    >>> from diffusers import LMSDiscreteScheduler

    >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    >>> pipeline.scheduler = scheduler
    ```
    """
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    resume_download = kwargs.pop("resume_download", False)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop(token_arg_name, None)
    revision = kwargs.pop("revision", None)
    from_flax = kwargs.pop("from_flax", False)
    torch_dtype = kwargs.pop("torch_dtype", None)
    custom_pipeline = kwargs.pop("custom_pipeline", None)
    custom_revision = kwargs.pop("custom_revision", None)
    provider = kwargs.pop("provider", None)
    sess_options = kwargs.pop("sess_options", None)
    device_map = kwargs.pop("device_map", None)
    max_memory = kwargs.pop("max_memory", None)
    offload_folder = kwargs.pop("offload_folder", None)
    offload_state_dict = kwargs.pop("offload_state_dict", False)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
    variant = kwargs.pop("variant", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    use_onnx = kwargs.pop("use_onnx", None)
    load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

    print("In our loading pipeline")
    # 1. Download the checkpoints and configs
    # use snapshot download here to get it working from from_pretrained
    if not os.path.isdir(pretrained_model_name_or_path):
        cached_folder = cls.download(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            from_flax=from_flax,
            use_safetensors=use_safetensors,
            use_onnx=use_onnx,
            custom_pipeline=custom_pipeline,
            custom_revision=custom_revision,
            variant=variant,
            load_connected_pipeline=load_connected_pipeline,
            **kwargs,
        )
    else:
        cached_folder = pretrained_model_name_or_path

    config_dict = cls.load_config(cached_folder)

    # pop out "_ignore_files" as it is only needed for download
    config_dict.pop("_ignore_files", None)

    # 2. Define which model components should load variants
    # We retrieve the information by matching whether variant
    # model checkpoints exist in the subfolders
    model_variants = {}
    if variant is not None:
        for folder in os.listdir(cached_folder):
            folder_path = os.path.join(cached_folder, folder)
            is_folder = os.path.isdir(folder_path) and folder in config_dict
            variant_exists = is_folder and any(
                p.split(".")[1].startswith(variant) for p in os.listdir(folder_path)
            )
            if variant_exists:
                model_variants[folder] = variant

    # 3. Load the pipeline class, if using custom module then load it from the hub
    # if we load from explicit class, let's use it
    pipeline_class = _get_pipeline_class(
        cls,
        config_dict,
        load_connected_pipeline=load_connected_pipeline,
        custom_pipeline=custom_pipeline,
        cache_dir=cache_dir,
        revision=custom_revision,
    )

    # DEPRECATED: To be removed in 1.0.0
    if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
        version.parse(config_dict["_diffusers_version"]).base_version
    ) <= version.parse("0.5.1"):
        from diffusers import (
            StableDiffusionInpaintPipeline,
            StableDiffusionInpaintPipelineLegacy,
        )

        pipeline_class = StableDiffusionInpaintPipelineLegacy

        deprecation_message = (
            "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
            f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
            " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
            " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
            f" checkpoint {pretrained_model_name_or_path} to the format of"
            " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
            " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
        )
        deprecate(
            "StableDiffusionInpaintPipelineLegacy",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )

    # 4. Define expected modules given pipeline signature
    # and define non-None initialized modules (=`init_kwargs`)

    # some modules can be passed directly to the init
    # in this case they are already instantiated in `kwargs`
    # extract them here
    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

    init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(
        config_dict, **kwargs
    )

    # define init kwargs and make sure that optional component modules are filtered out
    init_kwargs = {
        k: init_dict.pop(k)
        for k in optional_kwargs
        if k in init_dict and k not in pipeline_class._optional_components
    }
    init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

    # remove `null` components
    def load_module(name, value):
        if value[0] is None:
            return False
        if name in passed_class_obj and passed_class_obj[name] is None:
            return False
        return True

    init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

    # Special case: safety_checker must be loaded separately when using `from_flax`
    if (
        from_flax
        and "safety_checker" in init_dict
        and "safety_checker" not in passed_class_obj
    ):
        raise NotImplementedError(
            "The safety checker cannot be automatically loaded when loading weights `from_flax`."
            " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
            " separately if you need it."
        )

    # 5. Throw nice warnings / errors for fast accelerate loading
    if len(unused_kwargs) > 0:
        logger.warning(
            f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
        )

    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
            " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
            " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
            " install accelerate\n```\n."
        )

    if device_map is not None and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `device_map=None`."
        )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    if low_cpu_mem_usage is False and device_map is not None:
        raise ValueError(
            f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
            " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
        )

    # import it here to avoid circular import
    from diffusers import pipelines

    # 6. Load each module in the pipeline
    for name, (library_name, class_name) in tqdm(
        init_dict.items(), desc="Loading pipeline components..."
    ):
        # 6.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
        if class_name.startswith("Flax"):
            class_name = class_name[4:]

        # 6.2 Define all importable classes
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = ALL_IMPORTABLE_CLASSES
        loaded_sub_model = None

        # 6.3 Use passed sub model or load class_name from library_name
        if name in passed_class_obj:
            # if the model is in a pipeline module, then we load it from the pipeline
            # check that passed_class_obj has correct parent class
            maybe_raise_or_warn(
                library_name,
                library,
                class_name,
                importable_classes,
                passed_class_obj,
                name,
                is_pipeline_module,
            )

            loaded_sub_model = passed_class_obj[name]
        else:
            # load sub model
            loaded_sub_model = load_sub_model(
                library_name=library_name,
                class_name=class_name,
                importable_classes=importable_classes,
                pipelines=pipelines,
                is_pipeline_module=is_pipeline_module,
                pipeline_class=pipeline_class,
                torch_dtype=torch_dtype,
                provider=provider,
                sess_options=sess_options,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                model_variants=model_variants,
                name=name,
                from_flax=from_flax,
                variant=variant,
                low_cpu_mem_usage=low_cpu_mem_usage,
                cached_folder=cached_folder,
            )
            # logger.info(
            #    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
            # )

        init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

    if pipeline_class._load_connected_pipes and os.path.isfile(
        os.path.join(cached_folder, "README.md")
    ):
        modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
        connected_pipes = {
            prefix: getattr(modelcard.data, prefix, [None])[0]
            for prefix in CONNECTED_PIPES_KEYS
        }
        load_kwargs = {
            "cache_dir": cache_dir,
            "resume_download": resume_download,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            token_arg_name: use_auth_token,
            "revision": revision,
            "torch_dtype": torch_dtype,
            "custom_pipeline": custom_pipeline,
            "custom_revision": custom_revision,
            "provider": provider,
            "sess_options": sess_options,
            "device_map": device_map,
            "max_memory": max_memory,
            "offload_folder": offload_folder,
            "offload_state_dict": offload_state_dict,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "variant": variant,
            "use_safetensors": use_safetensors,
        }

        def get_connected_passed_kwargs(prefix):
            connected_passed_class_obj = {
                k.replace(f"{prefix}_", ""): w
                for k, w in passed_class_obj.items()
                if k.split("_")[0] == prefix
            }
            connected_passed_pipe_kwargs = {
                k.replace(f"{prefix}_", ""): w
                for k, w in passed_pipe_kwargs.items()
                if k.split("_")[0] == prefix
            }

            connected_passed_kwargs = {
                **connected_passed_class_obj,
                **connected_passed_pipe_kwargs,
            }
            return connected_passed_kwargs

        connected_pipes = {
            prefix: DiffusionPipeline.from_pretrained(
                repo_id, **load_kwargs.copy(), **get_connected_passed_kwargs(prefix)
            )
            for prefix, repo_id in connected_pipes.items()
            if repo_id is not None
        }

        for prefix, connected_pipe in connected_pipes.items():
            # add connected pipes to `init_kwargs` with <prefix>_<component_name>, e.g. "prior_text_encoder"
            init_kwargs.update(
                {
                    "_".join([prefix, name]): component
                    for name, component in connected_pipe.components.items()
                }
            )

    # 7. Potentially add passed objects if expected
    missing_modules = set(expected_modules) - set(init_kwargs.keys())
    passed_modules = list(passed_class_obj.keys())
    optional_modules = pipeline_class._optional_components
    if len(missing_modules) > 0 and missing_modules <= set(
        passed_modules + optional_modules
    ):
        for module in missing_modules:
            init_kwargs[module] = passed_class_obj.get(module, None)
    elif len(missing_modules) > 0:
        passed_modules = (
            set(list(init_kwargs.keys()) + list(passed_class_obj.keys()))
            - optional_kwargs
        )
        raise ValueError(
            f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
        )

    # 8. Instantiate the pipeline
    model = pipeline_class(**init_kwargs)

    # 9. Save where the model was instantiated from
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    return model


ORIGIN_DIFFUDION_PIPELINE = None


def enable_deep_cache_pipeline():
    global ORIGIN_DIFFUDION_PIPELINE
    if ORIGIN_DIFFUDION_PIPELINE is None:
        ORIGIN_DIFFUDION_PIPELINE = diffusers.DiffusionPipeline.from_pretrained
        diffusers.DiffusionPipeline.from_pretrained = classmethod(from_pretrained)


def disable_deep_cache_pipeline():
    if ORIGIN_DIFFUDION_PIPELINE is None:
        return
    diffusers.DiffusionPipeline.from_pretrained = ORIGIN_DIFFUDION_PIPELINE


__all__ = [
    "enable_deep_cache_pipeline",
    "disable_deep_cache_pipeline",
]
