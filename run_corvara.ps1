$r = "W:\SoLoMon\Siti\Corvara\DEM\LiDAR_HELICA_2019_2022\Elaborazioni_UNIMORE\IRIS_corvara_2022\Elaborazione_HSD_NCC\processing_2\HSD_2019.tiff"
$t = "W:\SoLoMon\Siti\Corvara\DEM\LiDAR_HELICA_2019_2022\Elaborazioni_UNIMORE\IRIS_corvara_2022\Elaborazione_HSD_NCC\processing_2\HSD_2021.tiff"

.\wp32\Scripts\activate

python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing minmax --output CV_2019-2021_ski-ilk.minmax.tiff
python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing zscore --output CV_2019-2021_ski-ilk.zscore.tiff
python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing clahe --output CV_2019-2021_ski-ilk.clahe.tiff
python -m ot.main -r $r -t $t -ot SKIOFILK --preprocessing lognorm --output CV_2019-2021_ski-ilk.lognorm.tiff

python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing minmax --output CV_2019-2021_ski-tvl1.minmax.tiff
python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing zscore --output CV_2019-2021_ski-tvl1.zscore.tiff
python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing clahe --output CV_2019-2021_ski-tvl1.clahe.tiff
python -m ot.main -r $r -t $t -ot SKIOFTVL1 --preprocessing lognorm --output CV_2019-2021_ski-tvl1.lognorm.tiff

python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing lognorm --output CV_2019-2021_opencv.lognorm.tiff
python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing minmax --output CV_2019-2021_opencv.minmax.tiff
python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing zscore --output CV_2019-2021_opencv.zscore.tiff
python -m ot.main -r $r -t $t -ot OPENCVOF --preprocessing clahe --output CV_2019-2021_opencv.clahe.tiff