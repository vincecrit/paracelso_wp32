@ECHO OFF

SET SLCPREP0="output\D_CL_S1SLC_NOSF_168_20240720T045052590968.tif"
SET SLCPREP1="output\D_CL_S1SLC_NOSF_168_20241223T045050567992.tif"
SET SLCPREP2="output\D_CL_S1SLC_NOSF_168_20250128T045047807536.tif"

ECHO ATTIVO AMBIENTE
CALL .\wp32\Scripts\activate.bat

ECHO OFFSET TRACKING (OpenCV)
python -m ot.main -ot OPENCVOF ^
   --reference %SLCPREP0% --target %SLCPREP1% ^
   --winsize 1 --poly_n 2 --poly_sigma 1 --levels 0 ^
   -o output\OT_OPENCVOF.tif

ECHO OFFSET TRACKING (scikit-image ILV)
python -m ot.main -ot SKIOFILV ^
   --reference %SLCPREP0% --target %SLCPREP1% ^
   --radius 1 --gaussian --prefilter ^
   -o output\OT_SKIOFILV.tif

ECHO OFFSET TRACKING (scikit-image TV-L1)
python -m ot.main -ot SKIOFTVL1 ^
   --reference %SLCPREP1% --target %SLCPREP2% ^
   -o output\OT_SKIOFTVL1.tif