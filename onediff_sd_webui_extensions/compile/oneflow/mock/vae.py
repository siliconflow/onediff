from modules.sd_vae_approx import VAEApprox

from onediff.infer_compiler.backends.oneflow.transform import proxy_class


# Prevent re-importing modules.shared, which incorrectly initializes all its variables.
class VAEApproxOflow(proxy_class(VAEApprox)):
    pass


torch2oflow_class_map = {
    VAEApprox: VAEApproxOflow,
}
