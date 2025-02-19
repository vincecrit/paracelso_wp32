@ECHO OFF

SET SLCPREP0="output\D_aoi_calita_S1SLC_NOSFB3_168_20241223T045050567992.tif"
SET SLCPREP1="output\D_aoi_calita_S1SLC_NOSFB3_168_20250128T045047807536.tif"
SET SLCPREP2="output\D_CL_S1SLC_NOSF_168_20250128T045047807536.tif"

SET REF="W:\SoLoMon\Matlab\Input\2024_03_22_BALDIOLA_HSD10cm_CUT.tif"
SET TAR="W:\SoLoMon\Matlab\Input\2024_05_23_BALDIOLA_HSD10cm_CUT.tif"

ECHO ATTIVO AMBIENTE
CALL .\wp32\Scripts\activate.bat

@REM ECHO OFFSET TRACKING (OpenCV)
@REM python -m ot.main -ot OPENCVOF ^
@REM    --reference %REF% --target %TAR% ^
@REM    --winsize 10 --poly_n 5 --poly_sigma 1.1 --levels 1 ^
@REM    -o output\OT_OPENCVOF_BLD_20240322-20240523.tif

@REM ECHO OFFSET TRACKING (scikit-image ILV)
@REM python -m ot.main -ot SKIOFILV ^
@REM    --reference %REF% --target %TAR% ^
@REM    --radius 10 --gaussian --prefilter ^
@REM    -o output\OT_SKIOFILV_BLD_20240322-20240523.tif

REM ECHO OFFSET TRACKING (scikit-image TV-L1)
REM python -m ot.main -ot SKIOFTVL1 ^
REM    --reference %REF% --target %TAR% ^
REM    -o output\OT_SKIOFTVL1_BLD_20240322-20240523.tif

python -m ot.main -ot SKIPCCV -r %SLCPREP0% -t %SLCPREP1% -o "output/test.tif"