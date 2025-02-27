@ECHO OFF
SET REF="C:\Users\vcritell\pymods\paracelso_wp32\output\D_aoi_calita_S1SLC_DFLTB3_168_20241223T045050567992.tif"
SET TAR="C:\Users\vcritell\pymods\paracelso_wp32\output\D_aoi_calita_S1SLC_DFLTB3_168_20250128T045047807536.tif"
@REM SET REF="W:\SoLoMon\Siti\Corvara\DEM\LiDAR_HELICA_2019_2022\Elaborazioni_UNIMORE\IRIS_corvara_2022\Elaborazione_HSD_NCC\processing_2\HSD_2019.tiff"
@REM SET TAR="W:\SoLoMon\Siti\Corvara\DEM\LiDAR_HELICA_2019_2022\Elaborazioni_UNIMORE\IRIS_corvara_2022\Elaborazione_HSD_NCC\processing_2\HSD_2021.tiff"

ECHO ATTIVO AMBIENTE
CALL .\wp32\Scripts\activate.bat

@REM ECHO OFFSET TRACKING (OpenCV)
@REM python -m ot.main -ot OPENCVOF ^
@REM    --reference %REF% --target %TAR% ^
@REM    --winsize 8 ^
@REM    -o output\OT_OPENCVOF_corvara19_21.tif

@REM ECHO OFFSET TRACKING (OpenCV)
@REM python -m ot.main -ot OPENCVOF ^
@REM    --reference %REF% --target %TAR% ^
@REM    --winsize 8 --zscore ^
@REM    -o output\OT_OPENCVOF.ZSCORE_corvara19_21.tif

@REM ECHO OFFSET TRACKING (SKIOFTVL1)
@REM python -m ot.main -ot SKIOFTVL1 ^
@REM    --reference %REF% --target %TAR% ^
@REM    --radius 8 ^
@REM    -o output\OT_SKIOFTVL1_corvara19_21.tif

ECHO OFFSET TRACKING (SKIOFTVL1)
python -m ot.main -ot SKIOFTVL1 ^
   --reference %REF% --target %TAR% ^
   --radius 8 --zscore ^
   -o output\OT_SKIOFTVL1.ZSCORE_corvara19_21.tif

@REM ECHO OFFSET TRACKING (SKIOFILK)
@REM python -m ot.main -ot SKIOFILK ^
@REM    --reference %REF% --target %TAR% ^
@REM    --radius 8 --clahe^
@REM    -o output\OT_SKIOFILK_CLAHE.corvara19_21.tif

ECHO OFFSET TRACKING (SKIPCCV)
python -m ot.main -ot SKIPCCV ^
   --reference %REF% --target %TAR% ^
   --winsize 5 --step_size 2 --clahe^
   -o output\OT_SKIPCCV_calita24_25.gpkg