Search.setIndex({docnames:["data_generator","datasets","index","models","modules","monitors","rate_controllers","sconce","sconce.data_generators","sconce.datasets","sconce.models","sconce.models.layers","sconce.monitors","sconce.rate_controllers","sconce.trainers","trainers","transforms"],envversion:53,filenames:["data_generator.rst","datasets.rst","index.rst","models.rst","modules.rst","monitors.rst","rate_controllers.rst","sconce.rst","sconce.data_generators.rst","sconce.datasets.rst","sconce.models.rst","sconce.models.layers.rst","sconce.monitors.rst","sconce.rate_controllers.rst","sconce.trainers.rst","trainers.rst","transforms.rst"],objects:{"sconce.data_generators":{DataGenerator:[0,0,1,""],ImageDataGenerator:[0,0,1,""],MultiClassImageDataGenerator:[0,0,1,""],SingleClassImageDataGenerator:[0,0,1,""],base:[8,4,0,"-"],image_data_generator:[8,4,0,"-"],image_mixin:[8,4,0,"-"],multi_class_image_data_generator:[8,4,0,"-"],single_class_image_data_generator:[8,4,0,"-"]},"sconce.data_generators.DataGenerator":{batch_size:[0,1,1,""],cuda:[0,2,1,""],dataset:[0,1,1,""],from_dataset:[0,3,1,""],num_samples:[0,1,1,""],real_dataset:[0,1,1,""],reset:[0,2,1,""]},"sconce.data_generators.MultiClassImageDataGenerator":{batch_size:[0,1,1,""],cuda:[0,2,1,""],dataset:[0,1,1,""],from_dataset:[0,3,1,""],get_class_df:[0,2,1,""],get_image_size_df:[0,2,1,""],num_channels:[0,1,1,""],num_samples:[0,1,1,""],plot_class_summary:[0,2,1,""],plot_image_size_summary:[0,2,1,""],real_dataset:[0,1,1,""],reset:[0,2,1,""]},"sconce.data_generators.SingleClassImageDataGenerator":{batch_size:[0,1,1,""],cuda:[0,2,1,""],dataset:[0,1,1,""],from_dataset:[0,3,1,""],from_image_folder:[0,3,1,""],from_torchvision:[0,3,1,""],get_class_df:[0,2,1,""],get_image_size_df:[0,2,1,""],num_channels:[0,1,1,""],num_samples:[0,1,1,""],plot_class_summary:[0,2,1,""],plot_image_size_summary:[0,2,1,""],real_dataset:[0,1,1,""],reset:[0,2,1,""]},"sconce.data_generators.base":{DataGenerator:[8,0,1,""]},"sconce.data_generators.base.DataGenerator":{batch_size:[8,1,1,""],cuda:[8,2,1,""],dataset:[8,1,1,""],from_dataset:[8,3,1,""],next:[8,2,1,""],num_samples:[8,1,1,""],preprocess:[8,2,1,""],real_dataset:[8,1,1,""],reset:[8,2,1,""]},"sconce.data_generators.image_mixin":{ImageMixin:[8,0,1,""],get_image_size:[8,5,1,""]},"sconce.data_generators.image_mixin.ImageMixin":{get_class_df:[8,2,1,""],get_image_size_df:[8,2,1,""],num_channels:[8,1,1,""],plot_class_summary:[8,2,1,""],plot_image_size_summary:[8,2,1,""]},"sconce.data_generators.multi_class_image_data_generator":{MultiClassImageDataGenerator:[8,0,1,""]},"sconce.data_generators.single_class_image_data_generator":{SingleClassImageDataGenerator:[8,0,1,""]},"sconce.data_generators.single_class_image_data_generator.SingleClassImageDataGenerator":{from_image_folder:[8,3,1,""],from_torchvision:[8,3,1,""]},"sconce.datasets":{CsvImageFolder:[1,0,1,""],csv_image_folder:[9,4,0,"-"]},"sconce.datasets.csv_image_folder":{CsvImageFolder:[9,0,1,""]},"sconce.datasets.csv_image_folder.CsvImageFolder":{found_extensions:[9,1,1,""],get_sample:[9,2,1,""],get_target:[9,2,1,""],num_classes:[9,1,1,""]},"sconce.models":{BasicAutoencoder:[3,0,1,""],BasicClassifier:[3,0,1,""],BasicConvolutionalAutoencoder:[3,0,1,""],MultilayerPerceptron:[3,0,1,""],WideResnetImageClassifier:[3,0,1,""],basic_autoencoder:[10,4,0,"-"],basic_classifier:[10,4,0,"-"],multilayer_perceptron:[10,4,0,"-"],wide_resnet_image_classifier:[10,4,0,"-"]},"sconce.models.BasicClassifier":{new_from_yaml_file:[3,3,1,""],new_from_yaml_filename:[3,3,1,""]},"sconce.models.MultilayerPerceptron":{new_from_yaml_file:[3,3,1,""],new_from_yaml_filename:[3,3,1,""]},"sconce.models.basic_autoencoder":{BasicAutoencoder:[10,0,1,""]},"sconce.models.basic_autoencoder.BasicAutoencoder":{calculate_loss:[10,2,1,""],decode:[10,2,1,""],encode:[10,2,1,""],forward:[10,2,1,""]},"sconce.models.basic_classifier":{BasicClassifier:[10,0,1,""]},"sconce.models.basic_classifier.BasicClassifier":{calculate_loss:[10,2,1,""],calculate_metrics:[10,2,1,""],forward:[10,2,1,""],freeze_batchnorm_layers:[10,2,1,""],layers:[10,1,1,""],new_from_yaml_file:[10,3,1,""],new_from_yaml_filename:[10,3,1,""],unfreeze_batchnorm_layers:[10,2,1,""]},"sconce.models.layers":{convolution2d_layer:[11,4,0,"-"],fully_connected_layer:[11,4,0,"-"]},"sconce.models.layers.convolution2d_layer":{Convolution2dLayer:[11,0,1,""]},"sconce.models.layers.convolution2d_layer.Convolution2dLayer":{forward:[11,2,1,""],out_height:[11,2,1,""],out_width:[11,2,1,""]},"sconce.models.layers.fully_connected_layer":{FullyConnectedLayer:[11,0,1,""]},"sconce.models.layers.fully_connected_layer.FullyConnectedLayer":{forward:[11,2,1,""]},"sconce.models.multilayer_perceptron":{MultilayerPerceptron:[10,0,1,""]},"sconce.models.multilayer_perceptron.MultilayerPerceptron":{calculate_loss:[10,2,1,""],calculate_metrics:[10,2,1,""],forward:[10,2,1,""],new_from_yaml_file:[10,3,1,""],new_from_yaml_filename:[10,3,1,""]},"sconce.models.wide_resnet_image_classifier":{AdaptiveAveragePooling2dLayer:[10,0,1,""],WideResnetBlock_3x3:[10,0,1,""],WideResnetGroup_3x3:[10,0,1,""],WideResnetImageClassifier:[10,0,1,""]},"sconce.models.wide_resnet_image_classifier.AdaptiveAveragePooling2dLayer":{forward:[10,2,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetBlock_3x3":{forward:[10,2,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetGroup_3x3":{forward:[10,2,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetImageClassifier":{calculate_loss:[10,2,1,""],calculate_metrics:[10,2,1,""],forward:[10,2,1,""]},"sconce.monitors":{CompositeMonitor:[5,0,1,""],DataframeMonitor:[5,0,1,""],LosswiseMonitor:[5,0,1,""],Monitor:[5,0,1,""],RingbufferMonitor:[5,0,1,""],StdoutMonitor:[5,0,1,""],base:[12,4,0,"-"],dataframe_monitor:[12,4,0,"-"],losswise_monitor:[12,4,0,"-"],ringbuffer_monitor:[12,4,0,"-"],stdout_monitor:[12,4,0,"-"]},"sconce.monitors.Monitor":{end_session:[5,2,1,""],start_session:[5,2,1,""],write:[5,2,1,""]},"sconce.monitors.base":{CompositeMonitor:[12,0,1,""],Monitor:[12,0,1,""]},"sconce.monitors.base.CompositeMonitor":{add_monitor:[12,2,1,""],end_session:[12,2,1,""],start_session:[12,2,1,""],write:[12,2,1,""]},"sconce.monitors.base.Monitor":{end_session:[12,2,1,""],start_session:[12,2,1,""],write:[12,2,1,""]},"sconce.monitors.dataframe_monitor":{DataframeMonitor:[12,0,1,""]},"sconce.monitors.dataframe_monitor.DataframeMonitor":{df:[12,1,1,""],from_file:[12,3,1,""],is_blacklisted:[12,2,1,""],plot:[12,2,1,""],plot_learning_rate_survey:[12,2,1,""],save:[12,2,1,""],start_session:[12,2,1,""],write:[12,2,1,""]},"sconce.monitors.losswise_monitor":{LosswiseMonitor:[12,0,1,""]},"sconce.monitors.losswise_monitor.LosswiseMonitor":{start_session:[12,2,1,""],write:[12,2,1,""]},"sconce.monitors.ringbuffer_monitor":{RingbufferMonitor:[12,0,1,""]},"sconce.monitors.ringbuffer_monitor.RingbufferMonitor":{mean:[12,2,1,""],movement_index:[12,1,1,""],start_session:[12,2,1,""],std:[12,2,1,""],write:[12,2,1,""]},"sconce.monitors.stdout_monitor":{StdoutMonitor:[12,0,1,""]},"sconce.monitors.stdout_monitor.StdoutMonitor":{start_session:[12,2,1,""],write:[12,2,1,""]},"sconce.rate_controllers":{CompositeRateController:[6,0,1,""],ConstantRateController:[6,0,1,""],CosineRateController:[6,0,1,""],ExponentialRateController:[6,0,1,""],LinearRateController:[6,0,1,""],RateController:[6,0,1,""],StepRateController:[6,0,1,""],TriangleRateController:[6,0,1,""],base:[13,4,0,"-"],constant_rate_controller:[13,4,0,"-"],cosine_rate_controller:[13,4,0,"-"],exponential_rate_controller:[13,4,0,"-"],linear_rate_controller:[13,4,0,"-"],step_rate_controller:[13,4,0,"-"],triangle_rate_controller:[13,4,0,"-"]},"sconce.rate_controllers.RateController":{new_learning_rate:[6,2,1,""],start_session:[6,2,1,""]},"sconce.rate_controllers.base":{CompositeRateController:[13,0,1,""],RateController:[13,0,1,""]},"sconce.rate_controllers.base.CompositeRateController":{add_rate_controller:[13,2,1,""],new_learning_rate:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.base.RateController":{new_learning_rate:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.constant_rate_controller":{ConstantRateController:[13,0,1,""]},"sconce.rate_controllers.constant_rate_controller.ConstantRateController":{new_learning_rate:[13,2,1,""],reset_monitor:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.cosine_rate_controller":{CosineRateController:[13,0,1,""]},"sconce.rate_controllers.cosine_rate_controller.CosineRateController":{new_learning_rate:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.exponential_rate_controller":{ExponentialRateController:[13,0,1,""]},"sconce.rate_controllers.exponential_rate_controller.ExponentialRateController":{new_learning_rate:[13,2,1,""],should_continue:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.linear_rate_controller":{LinearRateController:[13,0,1,""]},"sconce.rate_controllers.linear_rate_controller.LinearRateController":{new_learning_rate:[13,2,1,""],should_continue:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.step_rate_controller":{StepRateController:[13,0,1,""]},"sconce.rate_controllers.step_rate_controller.StepRateController":{new_learning_rate:[13,2,1,""],start_session:[13,2,1,""]},"sconce.rate_controllers.triangle_rate_controller":{TriangleRateController:[13,0,1,""]},"sconce.rate_controllers.triangle_rate_controller.TriangleRateController":{new_learning_rate:[13,2,1,""],start_session:[13,2,1,""]},"sconce.trainer":{Trainer:[15,0,1,""]},"sconce.trainer.Trainer":{checkpoint:[15,2,1,""],load_model_state:[15,2,1,""],multi_train:[15,2,1,""],num_trainable_parameters:[15,1,1,""],restore:[15,2,1,""],save_model_state:[15,2,1,""],survey_learning_rate:[15,2,1,""],test:[15,2,1,""],train:[15,2,1,""]},"sconce.trainers":{AutoencoderTrainer:[15,0,1,""],ClassifierTrainer:[15,0,1,""],SingleClassImageClassifierTrainer:[15,0,1,""],autoencoder_trainer:[14,4,0,"-"],single_class_image_classifier_trainer:[14,4,0,"-"]},"sconce.trainers.AutoencoderTrainer":{checkpoint:[15,2,1,""],load_model_state:[15,2,1,""],multi_train:[15,2,1,""],num_trainable_parameters:[15,1,1,""],restore:[15,2,1,""],save_model_state:[15,2,1,""],survey_learning_rate:[15,2,1,""],test:[15,2,1,""],train:[15,2,1,""]},"sconce.trainers.SingleClassImageClassifierTrainer":{checkpoint:[15,2,1,""],load_model_state:[15,2,1,""],multi_train:[15,2,1,""],num_trainable_parameters:[15,1,1,""],plot_samples:[15,2,1,""],restore:[15,2,1,""],save_model_state:[15,2,1,""],survey_learning_rate:[15,2,1,""],test:[15,2,1,""],train:[15,2,1,""]},"sconce.trainers.autoencoder_trainer":{AutoencoderMixin:[14,0,1,""],AutoencoderTrainer:[14,0,1,""]},"sconce.trainers.autoencoder_trainer.AutoencoderMixin":{plot_input_output_pairs:[14,2,1,""],plot_latent_space:[14,2,1,""]},"sconce.trainers.single_class_image_classifier_trainer":{ClassifierTrainer:[14,0,1,""],SingleClassImageClassifierMixin:[14,0,1,""],SingleClassImageClassifierTrainer:[14,0,1,""]},"sconce.trainers.single_class_image_classifier_trainer.SingleClassImageClassifierMixin":{get_classification_accuracy:[14,2,1,""],get_confusion_matrix:[14,2,1,""],plot_confusion_matrix:[14,2,1,""],plot_samples:[14,2,1,""]},"sconce.transforms":{NHot:[16,0,1,""]},"sconce.utils":{Progbar:[7,0,1,""]},"sconce.utils.Progbar":{add:[7,2,1,""],update:[7,2,1,""]},sconce:{models:[3,4,0,"-"],trainer:[7,4,0,"-"],transforms:[7,4,0,"-"],utils:[7,4,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","attribute","Python attribute"],"2":["py","method","Python method"],"3":["py","classmethod","Python class method"],"4":["py","module","Python module"],"5":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:attribute","2":"py:method","3":"py:classmethod","4":"py:module","5":"py:function"},terms:{"0x7fb1fbd498d0":[5,12],"100x1x28x28":[0,8],"6th":[7,15],"abstract":8,"case":[5,12],"class":[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16],"float":[0,5,6,7,8,12,13,14,15],"function":[0,1,3,6,8,9,13],"import":[5,6,12,13],"int":[0,1,3,5,6,7,8,9,10,12,13,14,15,16],"new":[0,3,6,8,10,13,14,15],"return":[0,1,3,6,7,8,9,13,15,16],"true":[0,3,5,8,10,11,12,14,15],For:[0,3,7,8,15],The:[0,3,5,6,7,8,12,13,14,15],These:[7,15],Used:[14,15],Using:[5,6,12,13],__next__:[0,8],_input:[5,12],_output:[5,12],_target:[5,12],abc:[8,12,13,14],abov:[3,10],accept:[3,5,12],access:[5,12],accordingli:[0,14,15],activ:[3,10,11],actual:[14,15],adaptiveaveragepooling2dlay:10,add:7,add_monitor:12,add_rate_control:13,added:[6,13],addit:[5,12],adjust:[6,7,13,15],after:[5,6,7,12,13,15],align:[6,13],all:[5,6,7,12,13,15],allow:[1,5,6,7,9,12,13,15],along:[7,15],also:[7,15],analyz:[7,15],ani:[0,3,8],anoth:[7,15],api_kei:[5,12],arbitrari:3,arg:[0,8,12,13,14,15],argument:[0,3,7,8,15],around:[0,8],arrai:[7,16],autoencod:[3,10,14],autoencoder_train:[4,15],autoencodermixin:[14,15],autoencodertrain:[2,14],automat:[0,8],averag:7,back:[3,6,7,13,15],bar:7,barchart:[0,8],base:[0,3,4,5,6,7,9,10,11,14,15],basic:[2,3,10],basic_autoencod:4,basic_classifi:4,basicautoencod:[2,10],basicclassif:2,basicclassifi:[3,10],basicconvolutionalautoencod:2,batch:[7,15],batch_multipl:[7,15],batch_multipli:[5,6,7,12,13,15],batch_siz:[0,8],been:[0,5,12,14,15],befor:[0,6,7,8,13,15],begin:[0,6,8,13],behavior:[0,3,8],belong:[0,1,8,9,14,15],below:[3,14,15],between:[6,13],bewar:[14,15],binari:[3,10],blacklist:[5,12],bmp:[1,9],bool:[0,8,14,15],both:[0,8],built:[3,10],cache_result:[14,15],calcul:3,calculate_loss:[3,10],calculate_metr:[3,10],call:[5,6,7,8,12,13,14,15],callabl:[0,1,8,9],can:[1,3,5,6,7,9,10,12,13,14,15],capac:[5,12],caus:[14,15],channel:[0,3,8,10],charact:[1,9],checkpoint:[7,15],chosen:[7,15],cifar100:[0,8],cifar10:[0,8],class_to_idx:[1,9],classes_delimit:[1,9],classes_kei:[1,9],classifi:[3,10,14,15],classification_accuraci:[3,10],classifier_train:4,classifiertrain:[2,14],classmethod:[0,3,8,10,12],code:[0,14,15],collect:[6,13],column:[1,9,14,15],come:3,complet:[5,6,7,12,13,15],compos:[5,6,12,13],composit:[7,15],compositemonitor:[2,12],compositeratecontrol:[2,13],connect:[3,10],consist:[7,15],constant:[6,13],constant_rate_control:4,constantratecontrol:[2,13],constraint:3,construct:[3,10],constructor:[0,7,8,15],contain:[0,1,8,9],content:[3,10],continu:[0,14,15],control:[2,13],conv_channel:3,convert:[7,16],convolut:[3,10],convolution2d_lay:10,convolution2dlay:[3,10,11],convolutional_layer_attribut:[3,10],convolutional_layer_kwarg:[3,10],convolutional_layer_valu:[3,10],copi:[0,8],correctli:[3,10],cosin:[6,13],cosine_rate_control:4,cosineratecontrol:[2,7,13,15],cours:[6,13],cpu:[0,8],creat:[0,7,8,15],cross:[3,10],csv:[1,9],csv_delimit:[1,9],csv_image_fold:4,csv_path:[1,9],csvimagefold:[2,9],cuda:[0,8],current:7,cycl:[7,15],cycle_length:[7,15],cycle_multipli:[7,15],data:[2,5,6,7,8,9,12,13,14,15],data_gener:[0,4,7,14,15],data_load:[0,8],data_loc:[0,8],datafram:[0,8],dataframe_monitor:[4,5],dataframemonitor:[2,7,12,15],datagener:[2,7,8,15],dataload:[0,8],datasest:2,dataset:[0,1,3,4,7,8,10,14,15],dataset_class:[0,8],dataset_kwarg:[0,8],decod:[3,10],default_load:[1,9],defin:[3,5,6,7,12,13,15],densli:[3,10],depend:[0,7,8,15],deprec:[2,4],depth:[3,10],describ:[3,6,10,13],detail:[0,3,8,10],detect:[6,13],determin:[3,7,10,15],devic:[0,8],dict:[0,1,3,5,6,8,9,10,12,13],dictionari:[0,1,3,8,9,10],differ:[3,10],directli:[0,8],directori:[0,1,8,9],disk:[1,9],displai:7,divis:[3,10],download:[0,8],drop:[6,13],drop_factor:[6,13],dropout:[3,10,11],dure:[5,6,7,12,13,15],each:[0,1,3,8,9,10],earli:[7,15],ect:[6,7,13,15],effect:[7,15],els:7,encod:[3,7,10,16],end:[5,6,12,13],end_sess:[5,12],endlessli:[0,8],entropi:[3,10],epoch:[0,7,8,15],equal:[7,15],evalu:[5,6,12,13],everi:[0,8],exactli:[0,8],exampl:[0,3,7,8,10,15,16],expect:[3,5,6,7,12,13],exponenti:[6,13],exponential_rate_control:[4,7,15],exponentialratecontrol:[2,7,13,15],extens:[1,9],extern:[7,15],factor:[7,15],fall:[6,13,14,15],fals:[0,8,10,11],fashionmnist:[0,3,8,10],faster:[14,15],few:3,field:[1,9],fig:12,figsiz:[12,14],figur:[14,15],figure_kwarg:12,figure_width:[14,15],file:[1,3,7,9,10,15],filenam:[1,3,7,9,10,12,15],filename_kei:[1,9],find:[14,15],first:[0,7,8,15],floattensor:[0,8],folder:[0,1,8,9],follow:[3,6,10,13],form:3,forward:[3,10,11],found:[1,9],found_extens:9,fraction:[0,3,5,6,8,10,12,13],freeze_batchnorm_lay:10,from:[0,1,3,5,6,7,8,9,10,12,13,15],from_dataset:[0,8],from_fil:12,from_image_fold:[0,8],from_torchvis:[0,8],frome:[7,15],fulli:[3,10],fully_connected_lay:10,fully_connected_layer_attribut:[3,10],fully_connected_layer_kwarg:[3,10],fully_connected_layer_valu:[3,10],fullyconnectedlay:[3,10,11],futur:[5,12],gain:[5,12],gener:[2,8,14,15],get:3,get_class_df:[0,8],get_classification_accuraci:14,get_confusion_matrix:14,get_image_s:8,get_image_size_df:[0,8],get_sampl:9,get_target:9,given:[1,7,9,14,15],gpu:[0,8],gradient:3,greater:[7,15],group:[6,13],handi:[8,14,15],has:[0,5,6,7,12,13,14,15],have:[0,5,8,12],header:[1,9],heatmap_kwarg:14,height:[3,10,14,15],hidden:[3,10],hidden_s:[3,10],higest:[14,15],highest:[14,15],histori:12,hot:[7,16],how:[0,3,7,8,10,15],human:[1,9],imag:[0,1,3,8,9,10,14,15],image_channel:[3,10],image_data_gener:4,image_height:[3,10,14,15],image_mixin:[0,4],image_nam:[1,9],image_width:[3,10],imagedatagener:[2,8],imagefold:[0,8],imagemixin:[0,8],imagenet:[14,15],implement:[6,13],in_channel:[10,11],in_height:11,in_siz:11,in_width:11,inch:[14,15],includ:[3,5,6,12,13],increas:[7,15],index:[2,7,9,14,15],indic:[1,7,9,16],individu:[0,8],inf:[3,5,6,7,10,12,13,15],infer:[7,15],inplace_activ:[10,11],input:[0,3,7,8,10,15],instanti:[0,8],instead:[0,8],integ:[3,10],interfac:[5,6,12,13],interv:7,is_blacklist:12,iter:[0,5,6,7,8,12,13],its:[1,6,9,13],jpeg:[1,9],jpg:[1,9],just:[3,5,6,12,13],keep:[14,15],kei:[0,3,5,6,8,12,13],kept:[7,15],kernel_s:[3,10,11],keyword:[0,3,7,8,15],kwarg:[0,5,7,8,10,12,14,15],label:[3,10],larg:[0,7,8,14,15],latent:[3,10,14],latent_s:[3,10],later:[7,15],layer:[3,10],layer_attribut:[3,10],layer_kwarg:[3,10],layer_valu:[3,10],learn:[5,6,7,12,13,15],learning_r:[5,6,12,13],least:3,len:[0,8],length:[3,7,15],like:[0,3,5,6,8,10,12,13,14,15],limit:[5,6,12,13],linear_rate_control:4,linearli:[6,13],linearratecontrol:[2,7,13,15],list:[1,3,7,9,10,16],live:[0,8],load:[0,1,8,9],load_model_st:[7,15],loader:[0,1,8,9],loader_kwarg:[0,8],locat:[0,7,8,15],longtensor:[0,8],look:[0,8],loop:[7,15],loss:[3,5,6,7,10,12,13,15],loss_kei:[6,13],losswise_monitor:[4,5],losswisemonitor:[2,12],lowest:[14,15],made:3,mai:[0,3,5,6,8,12,13],make:[7,14,15],mani:[0,3,8,10],map:[1,9],matplotlib:[14,15],max_graph:[5,12],max_learning_r:[6,7,13,15],maximum:[7,15],mean:12,mediumseagreen:12,memori:[0,7,8,14,15],metadata:[5,12],method:[0,3,5,7,8,12,14,15],metric:[3,5,6,7,10,12,13],metric_nam:[5,12],min_graph:[5,12],min_learning_r:[6,7,13,15],minimum:[7,15],mixin:8,mnist:[0,3,7,8,10,15],mode:[7,15],model:[2,4,7,14,15],modifi:3,modul:[2,3,4,15],monitor:[2,4,7,14,15],more:[0,3,5,6,7,8,10,12,13,15],move:[6,13],movement_index:12,movement_kei:[6,13],movement_threshold:[6,13],movement_window:[6,13],much:[0,8],multi:[3,10],multi_class_image_data_gener:4,multi_train:[7,15],multiclassimagedatagener:[2,8],multilayer_perceptron:4,multilayerperceptron:[2,10],multipl:[6,7,13,15],multipli:[7,15],must:[3,5,6,12,13],name:[1,5,7,9,12],network:[3,10],never:[7,15],new_from_yaml_fil:[3,10],new_from_yaml_filenam:[3,10],new_learning_r:[6,13],next:[0,6,8,13],nhot:[1,2,7,9],none:[0,1,3,5,6,7,8,9,10,12,13,14,15],note:[3,14,15],now:[0,14,15],num_block:10,num_categori:[3,10],num_channel:[0,8],num_class:9,num_col:[14,15],num_cycl:[7,15],num_drop:[6,13],num_epoch:[7,15],num_sampl:[0,8,14,15],num_step:[5,6,12,13],num_trainable_paramet:[7,15],num_work:[0,8],number:[0,3,5,6,7,8,10,12,13,14,15],object:[3,5,7,8,10,12,15],occur:[7,15],often:[7,15],one:[0,7,8,14,15],onli:[6,7,13,15],oper:[5,12],optim:[3,6,7,13,14,15],optino:[1,9],option:[1,5,6,7,9,12,13,15],orchestr:[7,15],order:[6,13,14,15],ordereddict:[6,13],origin:[0,8],otehr:3,other:[3,5,7,12,13,15],otherwis:[0,8],out:[14,15],out_channel:[3,10,11],out_height:11,out_siz:[3,10,11],out_width:11,output:[3,5,6,10,12,13],output_s:10,over:[6,7,13],packag:4,pad:[3,10,11],panda:[0,8],paper:[3,10],param:[5,12],paramet:[0,1,3,5,6,7,8,9,10,12,13,14,15,16],parameter_group:[6,13],pass:[0,3,5,6,7,8,12,13,15],path:[0,1,3,7,8,9,10,15],per:[14,15],perceptron:[3,10],pgm:[1,9],pil:[0,8],pin:[0,8],pin_memori:[0,8],pixel:[3,10],pleas:[0,14,15],plot:[0,7,8,12,14,15],plot_class_summari:[0,8],plot_confusion_matrix:14,plot_image_size_summari:[0,8],plot_input_output_pair:14,plot_kwarg:12,plot_latent_spac:14,plot_learning_rate_survei:12,plot_sampl:[14,15],png:[1,9],posit:[3,10],possibl:[5,6,12,13],ppm:[1,9],practic:[7,15],preactiv:[10,11],predict:[3,10,14,15],predicted_class:[14,15],preprocess:8,previou:[7,15],previous:[7,15],produc:[5,12],progbar:7,progbar_kwarg:[5,12],progress:7,propag:[3,7,15],provid:[0,8],put:[0,8],python:3,pytorch:[0,7,8,15],randomcrop:[1,9],rate:[2,5,7,12,13,15],rate_control:[4,6,7,14,15],rate_controller_class:[7,15],rate_controller_kwarg:[7,15],rate_monitor:[7,15],ratecontrol:[2,7,13,15],reach:[0,8],read:[1,3,9,10],readabl:[1,9],real_dataset:[0,8],record:[5,7,12,15],refer:[0,2,8],rel:[7,15],relat:[3,10],remain:[3,6,10,13],remov:[0,14,15],represent:[3,10,14],reset:[0,8],reset_monitor:13,reshuffl:[0,8],resnet:[3,10],restor:[7,15],restrict:3,result:[14,15],retain:[7,15],ringbuffer_monitor:[4,5],ringbuffermonitor:[2,12],rise:[6,7,13,14,15],root:[0,1,8,9],run:[7,14,15],sampl:[1,7,9,14,15],satisfi:3,save:[7,12,15],save_model_st:[7,15],scale:[6,13],scatter:[0,8],sconc:[0,1,3,5,6,15,16],score:[14,15],screen:7,second:7,see:[0,3,5,6,7,8,10,12,13,14,15],self:[7,15],semi:7,sent:[7,15],separ:[1,9],session:[5,6,7,12,13,15],set:[0,8],shift:[6,13],should:[0,3,6,7,8,10,13],should_continu:13,show:[0,8],shuffl:[0,8],silent:7,simul:[7,15],singl:[5,6,12,13,14,15],single_class_image_classifier_train:[4,15],single_class_image_data_gener:4,singleclassimageclassifiermixin:[14,15],singleclassimageclassifiertrain:[2,14],singleclassimagedatagener:[2,8,14,15],size:[0,7,8,14,15,16],skip_first:12,smooth_window:12,some:[3,6,8,10,13,14,15],soon:[0,14,15],sort_bi:[14,15],sourc:[0,1,3,5,6,7,8,9,10,11,12,13,14,15,16],specif:[14,15],specifi:[0,8],start:[0,3,5,6,8,12,13],start_sess:[5,6,12,13],state:[7,15],stateful_metr:7,std:12,stdout_monitor:[4,5],stdoutmonitor:[2,7,12,15],step:[5,6,7,12,13],step_rate_control:4,stepratecontrol:[2,13],stop:[6,7,13,15],stop_factor:[6,7,13,15],store:[0,8],str:[5,12],stride:[3,10,11],string:[1,7,9,14,15],subclass:[3,7,15],submodul:4,subpackag:4,subprocess:[0,8],subsequ:[14,15],subset:[0,8],support:[0,8,14,15],survei:[7,15],survey_learning_r:[7,15],system:[0,7,8,14,15],tag:[1,5,9,12],take:[0,1,8,9],target:[0,1,3,7,8,9,10,15],target_transform:[1,9],task:[5,12],temporari:[0,7,8,15],tensor:[0,3,8],test:[0,5,7,8,12,15],test_color:12,test_data_gener:[7,14,15],test_loss:[5,6,12,13],test_to_train_ratio:[7,15],than:[0,7,8,15],them:[0,8],thi:[0,3,5,6,7,8,10,12,13,14,15],thin:[0,8],three:[3,10],through:[0,7,8,15],tif:[1,9],time:[6,7,13],titl:[12,14],togeth:[5,6,12,13],tomato:12,torch:[0,3,7,8,9,10,11,15],torchvis:[0,8],total:[3,7,10],totensor:[0,8],train:[0,5,6,7,8,12,13,14,15],trainabl:[7,15],trainer:[2,3,4,5,6,12,13],training_color:12,training_data_gener:[7,14,15],training_loss:[5,6,12,13],transform:[0,1,2,4,8,9],triangle_rate_control:4,triangleratecontrol:[2,13],true_class:[14,15],tupl:[0,7,8],two:[0,5,6,8,12,13],type:[6,7,8,13,15],typic:[7,15],underli:[0,7,8,15],unfreeze_batchnorm_lay:10,unknown:7,until:[7,15],updat:[0,5,6,7,12,13,14,15],usag:[7,15],use:[0,5,7,8,12,14,15],used:[0,1,3,5,6,7,8,9,12,13,15],uses:[3,5,10,12],using:[5,12],util:[4,9],val_loss:[5,12],valu:[0,3,6,7,8,10,13,15],value_for_last_step:7,variabl:[1,9],variou:[7,15],vector:[7,16],verbos:7,version:[0,1,8,9],visual:7,want:[5,6,12,13],were:[7,15],what:[6,13],when:[0,5,6,8,12,13,14,15],where:[0,1,3,6,7,8,9,10,13,14,15],which:[0,1,8,9],whole:[14,15],wide:[3,10],wide_resnet_image_classifi:4,widening_factor:[3,10],wideresnetblock_3x3:10,wideresnetgroup_3x3:10,wideresnetimageclassifi:[2,10],width:[3,7,10,14,15],with_batchnorm:[3,10,11],without:[1,7,9,15],work:[0,14,15],would:[7,15],wrap:[0,8],wrapper:[0,8],write:[5,12],x_in:[10,11],x_latent:10,yaml:[3,10],yaml_fil:[3,10],yaml_filenam:[3,10],yield:[0,7,8,15],you:[3,5,6,7,12,13,14,15],your:[0,14,15]},titles:["Data Generators","Datasests","Sconce Documentation","Models","sconce","Monitors","Rate Controllers","sconce package","sconce.data_generators package","sconce.datasets package","sconce.models package","sconce.models.layers package","sconce.monitors package","sconce.rate_controllers package","sconce.trainers package","Trainers","Transforms"],titleterms:{autoencoder_train:14,autoencodertrain:15,base:[8,12,13],basic:15,basic_autoencod:10,basic_classifi:10,basicautoencod:3,basicclassif:3,basicconvolutionalautoencod:3,classifier_train:14,classifiertrain:15,compositemonitor:5,compositeratecontrol:6,constant_rate_control:13,constantratecontrol:6,control:6,convolution2d_lay:11,cosine_rate_control:13,cosineratecontrol:6,csv_image_fold:9,csvimagefold:1,data:0,data_gener:8,dataframe_monitor:12,dataframemonitor:5,datagener:0,datasest:1,dataset:9,deprec:[0,8,14,15],document:2,exponential_rate_control:13,exponentialratecontrol:6,fully_connected_lay:11,gener:0,image_data_gener:8,image_mixin:8,imagedatagener:0,indic:2,layer:11,linear_rate_control:13,linearratecontrol:6,losswise_monitor:12,losswisemonitor:5,model:[3,10,11],modul:[7,8,9,10,11,12,13,14],monitor:[5,12],multi_class_image_data_gener:8,multiclassimagedatagener:0,multilayer_perceptron:10,multilayerperceptron:3,nhot:16,packag:[7,8,9,10,11,12,13,14],rate:6,rate_control:13,ratecontrol:6,ringbuffer_monitor:12,ringbuffermonitor:5,sconc:[2,4,7,8,9,10,11,12,13,14],single_class_image_classifier_train:14,single_class_image_data_gener:8,singleclassimageclassifiertrain:15,singleclassimagedatagener:0,stdout_monitor:12,stdoutmonitor:5,step_rate_control:13,stepratecontrol:6,submodul:[7,10],subpackag:[7,10],tabl:2,trainer:[7,14,15],transform:[7,16],triangle_rate_control:13,triangleratecontrol:6,util:7,wide_resnet_image_classifi:10,wideresnetimageclassifi:3}})