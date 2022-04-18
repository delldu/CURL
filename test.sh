# python3 main.py \
#     --inference_img_dirpath=./adobe5k_dpe/curl_example_test_input \
#     --checkpoint_filepath=./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt

python3 main.py \
    --inference_img_dirpath=./adobe5k_dpe/curl_example_test_input \
    --checkpoint_filepath=./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model_weights.pt
