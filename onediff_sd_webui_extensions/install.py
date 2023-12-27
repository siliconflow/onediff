import launch


def install():
    if not launch.is_installed("oneflow"):
        print("oneflow is not installed! Installing...")
        launch.run_pip(
            "install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu118"
        )
    if not launch.is_installed("onediff"):
        print("onediff is not installed! Installing...")
        launch.run_pip("install git+https://github.com/siliconflow/onediff.git")


install()
