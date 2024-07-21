def patch_input_adapter(in_args, in_kwargs):
    return in_args, in_kwargs


def online_quantize_model(
    model,
    input_args,
    input_kwargs,
    seed=1,
    inplace=True,
    module_selector=lambda x: x,
    quant_config=None,
    calibration_info=None,
):
    """Optimize the quantization pipeline.

    Returns:
        tuple: A tuple containing the quantized model and the quantization
        status.
    """

    from onediff_quant.quantization import (
        OnlineQuantModule,
        create_quantization_calculator,
    )

    if getattr(quant_config, "quantization_calculator", None):
        calculator = quant_config.quantization_calculator
    else:
        calculator = create_quantization_calculator(
            model,
            quant_config,
            module_selector,
            seed,
            calibration_info=calibration_info,
        )
    module = OnlineQuantModule(calculator, False, inplace=inplace)
    in_args, in_kwargs = patch_input_adapter(input_args, input_kwargs)
    quantized_model, info = module.quantize_with_calibration(*in_args, **in_kwargs)
    status = module.collect_quantization_status(model, info)
    for _, layer in quantized_model.named_modules():
        layer._disable_param_update = True

    return quantized_model, status


def quantize_and_deploy_wrapper(func):
    def wrapper(self: "DeployableModule", *args, **kwargs):
        torch_model = self._torch_module
        quant_config = self._deployable_module_quant_config
        if quant_config:
            torch_model, _ = online_quantize_model(
                torch_model,
                args,
                kwargs,
                module_selector=lambda x: x,
                quant_config=quant_config,
                inplace=True,
            )
            self._deployable_module_quant_config = None
        output = func(self, *args, **kwargs)
        return output

    return wrapper
