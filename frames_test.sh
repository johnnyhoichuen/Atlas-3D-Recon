 #iphone
#data="ust_conf_iphone"
##framesArray=(40 80 120 160 200 240 280 320 640 960 1280 1600 1920 2240 2560 2880 3196)
##framesArray=(10 20 30 40 50 100 150 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3196)

### opvs
#data="ust_conf3_icp_opvs"
##framesArray=(50 100 150 200 400 600 800 1000 1200 1400 1600 1800 1936)
#framesArray=(10 20 30 40)

## opvs filtered photographer
data="ust_conf3_icp_opvs"
#framesArray=(50 100 150 200 400 600 800 1000 1200 1400 1455)
framesArray=(10 20 30 40)

for num in ${framesArray[@]}; do
#  echo $num
  python3 automate.py --config data/"$data"/config.yaml --num_frames_inference $num
done