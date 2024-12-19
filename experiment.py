import fire

from exp import downscaling, figures

if __name__ == "__main__":
    fire.Fire({"predict": downscaling.run, "figures": figures})
