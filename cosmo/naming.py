import cosmo.utils as utils


class CSKFilename:

    _MM = dict(
        HI='Himage',
        PP='PingPong',
        WR='WideRegion',
        HR='HugeRegion',
        S2='Spotlight 2'
    )

    _PP = dict(
        HH="Horizontal Tx/Horizontal Rx",
        VV="Vertical Tx/ Vertical Rx",
        HV="Horizontal Tx/ Vertical Rx",
        VH="Vertical Tx/ Horizontal Rx",
        CO="Co-polar acquisition (HH/VV)",
        CH="Cross polar acquisition (HH/HV) with Horizontal Tx polarization",
        CV="Cross polar acquisition (VV/VH) with Vertical Tx polarization"
    )

    _s = dict(L='Left', R='Right')

    _o = dict(A='Ascending', D='Descending')

    _D = dict(F="Fast delivery mode", S="Standard delivery mode")

    _G = dict(N='ON', F='OFF')

    @classmethod
    def parse_filename(cls, filename: str):

        NAME = utils.StrChopper(filename)
        mission = NAME.chop(3)
        
        if not mission == 'CSG':
            raise ValueError("Non è un file CSK (II generazione)")

        i = NAME.chop(2)
        _ = NAME.chop(1)
        YYY_Z = NAME.chop(5)
        _ = NAME.chop(1)
        MM = NAME.chop(2)
        _ = NAME.chop(1)
        SS = NAME.chop(2)
        _ = NAME.chop(1)
        PP = NAME.chop(2)
        _ = NAME.chop(1)
        s = NAME.chop(1)
        o = NAME.chop(1)
        _ = NAME.chop(1)
        D = NAME.chop(1)
        G = NAME.chop(1)
        _ = NAME.chop(1)
        start = utils.str2dt(NAME.chop(14))
        _ = NAME.chop(1)
        end = utils.str2dt(NAME.chop(14))

        msg = f"""
        {filename}
        COSMO-SkyMed (I Generation)

        Satellite ID:            {i}
        Product Type:            {YYY_Z}
        Instrument mode:         {cls._MM[MM]}
        Swath:                   {SS}
        Polarization:            {cls._PP[PP]}
        Look side:               {cls._s[s]}
        Orbit Direction:         {cls._o[o]}
        Delivery mode:           {cls._D[D]}
        Selective Avaialability: {cls._G[G]}
        Sensing start time:      {start}
        Sensing stop time:       {end}
        """
        return msg


class CSGFilename:

    _MMM = dict(
        S2A='Spotlight 2A',
        S2B='Spotlight 2B',
        S2C='Spotlight 2C',
        D2R='Spotlight 1 optimized resolution',
        D2S='Spotlight 2 optimized swath',
        D2J='Spotlight 1 joined',
        OQR='Spotlight 1 operational QuadPol optimized resolution',
        OQS='Spotlight 2 operational QuadPol optimized swath',
        STR='Stripmap',
        SC1='ScanSAR 1',
        SC2='ScanSAR 2',
        PPS='PingPong',
        QPS='QuadPol'
    )

    _PP = dict(
        HH="Horizontal Tx/Horizontal Rx",
        VV="Vertical Tx/ Vertical Rx",
        HV="Horizontal Tx/ Vertical Rx",
        VH="Vertical Tx/ Horizontal Rx"
    )

    _Q = dict(
        D='Downlinked',
        P='Predicted',
        F='Filtered',
        R='Restituted'
    )

    _H = dict(
        N='North',
        S='South'
    )

    _S = dict(
        F='Full standard size',
        C='Cropped product'
    )

    _s = dict(L='Left', R='Right')

    _o = dict(A='Ascending', D='Descending')

    def _LL(s: str):
        assert s[0] == 'Z'

        s = s[1:]
        if int(s) in [*range(1, 61)]:
            descr = f'UTM Zone {s}'
        elif s == '00':
            descr = f'South Pole area'
        elif s == '61':
            descr = f'North Pole area'
        
        return descr
    
    def _AAA(s:str):
        if s[0] == 'N':
            descr = 'Not squinted data'
        elif s[0] == 'F':
            a = s[1]
            descr = f"Forward squint ({a}°)"
        elif s[0] == 'B':
            a = s[1]
            descr = f"Backward squint ({a}°)"
        return descr

    @classmethod
    def parse_filename(cls, filename: str):

        NAME = utils.StrChopper(filename)
        mission = NAME.chop(3)

        if not mission == 'CSG':
            raise ValueError("Non è un file CSK (II generazione)")

        _ = NAME.chop(1)
        i = NAME.chop(5)
        _ = NAME.chop(1)
        YYY_Z = NAME.chop(5)
        _ = NAME.chop(1)
        RR = NAME.chop(2)
        AA = NAME.chop(2)
        _ = NAME.chop(1)
        MMM = NAME.chop(3)
        _ = NAME.chop(1)
        SSS = NAME.chop(3)
        _ = NAME.chop(1)
        PP = NAME.chop(2)
        _ = NAME.chop(1)
        s = NAME.chop(1)
        o = NAME.chop(1)
        _ = NAME.chop(1)
        Q = NAME.chop(1)
        _ = NAME.chop(1)
        start = utils.str2dt(NAME.chop(14))
        _ = NAME.chop(1)
        end = utils.str2dt(NAME.chop(14))
        j = NAME.chop(1)
        _ = NAME.chop(1)
        S = NAME.chop(1)
        _ = NAME.chop(1)
        ll = NAME.chop(2)
        H = NAME.chop(1)
        _ = NAME.chop(1)
        ZLL = NAME.chop(3)
        _ = NAME.chop(1)
        AAA = NAME.chop(3)


        msg = f"""
        {filename}
        COSMO-SkyMed (II Generation)

        Satellite ID:              {i}
        Product Type:              {YYY_Z}
        Number of range looks:     {RR}
        Number of azimuth looks:   {RR}
        Instrument mode:           {cls._MMM[MMM]}
        Swath:                     {SSS}
        Polarization:              {cls._PP[PP]}
        Look side:                 {cls._s[s]}
        Orbit Direction:           {cls._o[o]}
        Orbital data quality:      {cls._Q[Q]}
        Sensing start time:        {start}
        Sensing stop time:         {end}
        File sequence ID:          {j}
        Product coverage:          {cls._S[S]}
        Latitude scene center:     {int(ll)}°
        Hemisphere:                {cls._H[H]}
        East-direction location:   {cls._LL(ZLL)}
        Squint angle scene center: {cls._AAA(AAA)}
        """
        return msg