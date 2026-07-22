def build_corruption(opt, log, corrupt_type=None):
    """Build corruption method for the given task type.

    For CT-specific tasks (kernel_conversion, denoising, multi_task),
    paired DICOM loading handles the degradation implicitly — no explicit
    corruption transform is needed.  Returns None in those cases.
    """
    if corrupt_type is None:
        corrupt_type = opt.corrupt

    # CT-specific tasks: degradation is handled by paired DICOM data loading.
    if corrupt_type in ('kernel_conversion', 'denoising', 'multi_task'):
        return None

    # Generic image restoration tasks (kept for compatibility with the base I2SB).
    elif 'inpaint' in corrupt_type:
        from .inpaint import build_inpaint_center, build_inpaint_freeform
        mask = corrupt_type.split("-")[1]
        assert mask in ["center", "freeform1020", "freeform2030"]
        if mask == "center":
            return build_inpaint_center(opt, log, mask)
        elif "freeform" in mask:
            return build_inpaint_freeform(opt, log, mask)

    elif 'jpeg' in corrupt_type:
        from .jpeg import build_jpeg
        quality_factor = int(corrupt_type.split("-")[1])
        return build_jpeg(log, quality_factor)

    elif 'sr4x' in corrupt_type:
        from .superresolution import build_sr4x
        sr_filter = corrupt_type.split("-")[1]
        assert sr_filter in ["pool", "bicubic"]
        return build_sr4x(opt, log, sr_filter, image_size=opt.image_size)

    elif 'blur' in corrupt_type:
        from .blur import build_blur
        kernel = corrupt_type.split("-")[1]
        assert kernel in ["uni", "gauss"]
        return build_blur(opt, log, kernel)

    else:
        raise RuntimeWarning(f"Unknown corruption: {corrupt_type}!")
