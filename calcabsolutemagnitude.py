from mk_ai.MKAI import MKAI

def calc_absolute_magnitude(img_path: str, url: bool = False) -> float:
    spectral_type, _ = MKAI().get_spectral_class(img_path, url)
    luminosity_class, _ = MKAI().get_luminosity_class(img_path, url)
    print(spectral_type, luminosity_class)
    if spectral_type == 'O':
        if luminosity_class == 'I':
            return -5
        elif luminosity_class == 'II':
            return -4
        elif luminosity_class == 'III':
            return -4
        elif luminosity_class == 'IV':
            return -3
        elif luminosity_class == 'V':
            return -5

    elif spectral_type == 'B':
        if luminosity_class == 'I':
            return -4
        elif luminosity_class == 'II':
            return -3
        elif luminosity_class == 'III':
            return -3
        elif luminosity_class == 'IV':
            return -2
        elif luminosity_class == 'V':
            return -4

    elif spectral_type == 'A':
        if luminosity_class == 'I':
            return -3
        elif luminosity_class == 'II':
            return -2
        elif luminosity_class == 'III':
            return -2
        elif luminosity_class == 'IV':
            return -1
        elif luminosity_class == 'V':
            return -3

    elif spectral_type == 'F':
        if luminosity_class == 'I':
            return -2
        elif luminosity_class == 'II':
            return -1
        elif luminosity_class == 'III':
            return -1
        elif luminosity_class == 'IV':
            return 0
        elif luminosity_class == 'V':
            return -2

    elif spectral_type == 'G':
        if luminosity_class == 'I':
            return -1
        elif luminosity_class == 'II':
            return 0
        elif luminosity_class == 'III':
            return 0
        elif luminosity_class == 'IV':
            return 1
        elif luminosity_class == 'V':
            return -1

    elif spectral_type == 'K':
        if luminosity_class == 'I':
            return 0
        elif luminosity_class == 'II':
            return 1
        elif luminosity_class == 'III':
            return 1
        elif luminosity_class == 'IV':
            return 2
        elif luminosity_class == 'V':
            return 0

    elif spectral_type == 'M':
        if luminosity_class == 'I':
            return 1
        elif luminosity_class == 'II':
            return 2
        elif luminosity_class == 'III':
            return 2
        elif luminosity_class == 'IV':
            return 3
        elif luminosity_class == 'V':
            return 1

if __name__ == "__main__":
    print(calc_absolute_magnitude("/Users/jminding/Documents/Research/Star Images/Almach.jpg"))