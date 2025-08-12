def get_parameters(case, model):
    # General parameters for all cases
    if model == 'Model2':
        Param = {
            't_c': 0.0078,      # Thickness over chord
            'lambda': 0.0341,   # NOTE unsure if this is correct, check with the model
            'c': 0.5,           # Chord length
            'c1': 0.5,           # Chord length
            'h': 1.656          # Height of the wind tunnel
        }

    elif model == 'V3':
        Param = {
            't_c': 0.1066,      # Thickness over chord
            'lambda': 0.0539,   # NOTE unsure if this is correct, check with the model
            'c': 0.51948,       # Chord length
            'c1': 0.51948,      # Chord length
            'h': 1.656          # Height of the wind tunnel
        }

    ## # --- Case-specific parameters for model2 ---
    if case == 'Model2_zz_0.05c_top_Re_1e6':
        Param.update({
            'M': 0.0872,
            'factor': 1.2280629777488508,
            'amin': -50,
            'amax': 13.1
        })
    elif case == 'Model2_no_zz_Re_5e5':
        Param.update({
            'factor': 1.22,
            # Add other case-specific parameters here if needed
        })
    # Add more cases as needed...

    # --- Case-specific parameters for V3 ---
    elif case == 'V3_no_zz_Re_1e6':
        Param.update({
            'factor': 1.2239225487723924,
        })
    elif case == 'V3_no_zz_Re_15e5':
        Param.update({
            'factor': 1.2249108236460178,
        })
    elif case == 'V3_no_zz_Re_5e5':
        Param.update({
            'factor': 1.253161865459855,
        })
    elif case == 'V3_small_zz_bottom_Re_1e6':
        Param.update({
            'factor': 1.2257203864836412,
        })
    elif case == 'V3_small_zz_bottom_Re_5e5':
        Param.update({
            'factor': 1.2350871551987999,
        })
    elif case == 'V3_zz_0.05c_top_Re_1e6':
        Param.update({
            'factor': 1.2324419580754797,
        })
    elif case == 'V3_zz_0.05c_top_Re_5e5':
        Param.update({
            'factor': 1.2344242289818768,
        })
    elif case == 'V3_zz_bottom_0.05c_top_Re_1e6':
        Param.update({
            'factor': 1.2260159930246814,
        })
    elif case == 'V3_zz_bottom_0.05c_top_Re_5e5':
        Param.update({
            'factor': 1.2280224580901005,
        })
    elif case == 'V3_bottom_0.03c_top_Re_1e6':
        Param.update({
            'factor': 1.230408768315289,
        })
    elif case == 'V3_bottom_45deg_0.03c_top_Re_1e6':
        Param.update({
            'factor': 1.2295807563284857,
        })
    elif case == 'V3_bottom_45_deg_Re_1e6':
        Param.update({
            'factor': 1.2269439189190063,
        })
    elif case == 'V3_bottom_45_deg_Re_5e5':
        Param.update({
            'M': 0.0872,
            'factor': 1.2382314456571446,
        })

    return Param
PARAM = get_parameters('V3_no_zz_Re_1e6', 'V3')