from modules.sd_vae_approx import VAEApprox

from onediff.infer_compiler.backends.oneflow.transform import proxy_class, register


class VAEApproxOflow(proxy_class(VAEApprox)):
    pass


torch2oflow_class_map = {
    VAEApprox: VAEApproxOflow,
}
