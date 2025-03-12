$r = "C:\Users\localadmin\paracelso_wp32\output\D_aoi_calita_S1SLC_DFLTB3_168_20241223T045050567992.tif"
$t = "C:\Users\localadmin\paracelso_wp32\output\D_aoi_calita_S1SLC_DFLTB3_168_20250128T045047807536.tif"

.\wp32\Scripts\activate

python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing minmax --output CL_2024-2025_ski-ilk.minmax.tiff
python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing zscore --output CL_2024-2025_ski-ilk.zscore.tiff
python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing clahe --output CL_2024-2025_ski-ilk.clahe.tiff
python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing lognorm --output CL_2024-2025_ski-ilk.lognorm.tiff

python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing minmax --output CL_2024-2025_ski-tvl1.minmax.tiff
python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing zscore --output CL_2024-2025_ski-tvl1.zscore.tiff
python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing clahe --output CL_2024-2025_ski-tvl1.clahe.tiff
python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing lognorm --output CL_2024-2025_ski-tvl1.lognorm.tiff

python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing lognorm --output CL_2024-2025_opencv.lognorm.tiff
python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing minmax --output CL_2024-2025_opencv.minmax.tiff
python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing zscore --output CL_2024-2025_opencv.zscore.tiff
python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing clahe --output CL_2024-2025_opencv.clahe.tiff