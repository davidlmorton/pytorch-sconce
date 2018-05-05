Search.setIndex({docnames:["data_generator","index","models","modules","monitors","rate_controllers","sconce","sconce.data_generators","sconce.models","sconce.models.layers","sconce.monitors","sconce.rate_controllers","sconce.trainers","trainers"],envversion:53,filenames:["data_generator.rst","index.rst","models.rst","modules.rst","monitors.rst","rate_controllers.rst","sconce.rst","sconce.data_generators.rst","sconce.models.rst","sconce.models.layers.rst","sconce.monitors.rst","sconce.rate_controllers.rst","sconce.trainers.rst","trainers.rst"],objects:{"sconce.data_generators":{DataGenerator:[0,0,1,""],ImageDataGenerator:[0,0,1,""],base:[7,4,0,"-"],image_data_generator:[7,4,0,"-"]},"sconce.data_generators.DataGenerator":{batch_size:[0,1,1,""],cuda:[0,2,1,""],dataset:[0,1,1,""],from_dataset:[0,3,1,""],from_pytorch:[0,3,1,""],num_samples:[0,1,1,""],real_dataset:[0,1,1,""],reset:[0,2,1,""]},"sconce.data_generators.ImageDataGenerator":{from_image_folder:[0,3,1,""],from_torchvision:[0,3,1,""],get_summary_df:[0,2,1,""],num_channels:[0,1,1,""],plot_label_summary:[0,2,1,""],plot_size_summary:[0,2,1,""]},"sconce.data_generators.base":{DataGenerator:[7,0,1,""]},"sconce.data_generators.base.DataGenerator":{batch_size:[7,1,1,""],cuda:[7,2,1,""],dataset:[7,1,1,""],from_dataset:[7,3,1,""],from_pytorch:[7,3,1,""],next:[7,2,1,""],num_samples:[7,1,1,""],preprocess:[7,2,1,""],real_dataset:[7,1,1,""],reset:[7,2,1,""]},"sconce.data_generators.image_data_generator":{ImageDataGenerator:[7,0,1,""],get_image_info:[7,5,1,""]},"sconce.data_generators.image_data_generator.ImageDataGenerator":{from_image_folder:[7,3,1,""],from_torchvision:[7,3,1,""],get_summary_df:[7,2,1,""],num_channels:[7,1,1,""],plot_label_summary:[7,2,1,""],plot_size_summary:[7,2,1,""]},"sconce.models":{BasicAutoencoder:[2,0,1,""],BasicClassifier:[2,0,1,""],BasicConvolutionalAutoencoder:[2,0,1,""],MultilayerPerceptron:[2,0,1,""],WideResnetImageClassifier:[2,0,1,""],basic_autoencoder:[8,4,0,"-"],basic_classifier:[8,4,0,"-"],multilayer_perceptron:[8,4,0,"-"],wide_resnet_image_classifier:[8,4,0,"-"]},"sconce.models.BasicClassifier":{new_from_yaml_file:[2,3,1,""],new_from_yaml_filename:[2,3,1,""]},"sconce.models.MultilayerPerceptron":{new_from_yaml_file:[2,3,1,""],new_from_yaml_filename:[2,3,1,""]},"sconce.models.basic_autoencoder":{BasicAutoencoder:[8,0,1,""]},"sconce.models.basic_autoencoder.BasicAutoencoder":{calculate_loss:[8,2,1,""],decode:[8,2,1,""],encode:[8,2,1,""],forward:[8,2,1,""]},"sconce.models.basic_classifier":{BasicClassifier:[8,0,1,""]},"sconce.models.basic_classifier.BasicClassifier":{calculate_loss:[8,2,1,""],calculate_metrics:[8,2,1,""],forward:[8,2,1,""],freeze_batchnorm_layers:[8,2,1,""],layers:[8,1,1,""],new_from_yaml_file:[8,3,1,""],new_from_yaml_filename:[8,3,1,""],unfreeze_batchnorm_layers:[8,2,1,""]},"sconce.models.layers":{convolution2d_layer:[9,4,0,"-"],fully_connected_layer:[9,4,0,"-"]},"sconce.models.layers.convolution2d_layer":{Convolution2dLayer:[9,0,1,""]},"sconce.models.layers.convolution2d_layer.Convolution2dLayer":{forward:[9,2,1,""],out_height:[9,2,1,""],out_width:[9,2,1,""]},"sconce.models.layers.fully_connected_layer":{FullyConnectedLayer:[9,0,1,""]},"sconce.models.layers.fully_connected_layer.FullyConnectedLayer":{forward:[9,2,1,""]},"sconce.models.multilayer_perceptron":{MultilayerPerceptron:[8,0,1,""]},"sconce.models.multilayer_perceptron.MultilayerPerceptron":{calculate_loss:[8,2,1,""],calculate_metrics:[8,2,1,""],forward:[8,2,1,""],new_from_yaml_file:[8,3,1,""],new_from_yaml_filename:[8,3,1,""]},"sconce.models.wide_resnet_image_classifier":{AdaptiveAveragePooling2dLayer:[8,0,1,""],WideResnetBlock_3x3:[8,0,1,""],WideResnetGroup_3x3:[8,0,1,""],WideResnetImageClassifier:[8,0,1,""]},"sconce.models.wide_resnet_image_classifier.AdaptiveAveragePooling2dLayer":{forward:[8,2,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetBlock_3x3":{forward:[8,2,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetGroup_3x3":{forward:[8,2,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetImageClassifier":{calculate_loss:[8,2,1,""],calculate_metrics:[8,2,1,""],forward:[8,2,1,""]},"sconce.monitors":{CompositeMonitor:[4,0,1,""],DataframeMonitor:[4,0,1,""],LosswiseMonitor:[4,0,1,""],Monitor:[4,0,1,""],RingbufferMonitor:[4,0,1,""],StdoutMonitor:[4,0,1,""],base:[10,4,0,"-"],dataframe_monitor:[10,4,0,"-"],losswise_monitor:[10,4,0,"-"],ringbuffer_monitor:[10,4,0,"-"],stdout_monitor:[10,4,0,"-"]},"sconce.monitors.Monitor":{end_session:[4,2,1,""],start_session:[4,2,1,""],write:[4,2,1,""]},"sconce.monitors.base":{CompositeMonitor:[10,0,1,""],Monitor:[10,0,1,""]},"sconce.monitors.base.CompositeMonitor":{add_monitor:[10,2,1,""],end_session:[10,2,1,""],start_session:[10,2,1,""],write:[10,2,1,""]},"sconce.monitors.base.Monitor":{end_session:[10,2,1,""],start_session:[10,2,1,""],write:[10,2,1,""]},"sconce.monitors.dataframe_monitor":{DataframeMonitor:[10,0,1,""]},"sconce.monitors.dataframe_monitor.DataframeMonitor":{df:[10,1,1,""],from_file:[10,3,1,""],is_blacklisted:[10,2,1,""],plot:[10,2,1,""],plot_learning_rate_survey:[10,2,1,""],save:[10,2,1,""],start_session:[10,2,1,""],write:[10,2,1,""]},"sconce.monitors.losswise_monitor":{LosswiseMonitor:[10,0,1,""]},"sconce.monitors.losswise_monitor.LosswiseMonitor":{start_session:[10,2,1,""],write:[10,2,1,""]},"sconce.monitors.ringbuffer_monitor":{RingbufferMonitor:[10,0,1,""]},"sconce.monitors.ringbuffer_monitor.RingbufferMonitor":{mean:[10,2,1,""],movement_index:[10,1,1,""],start_session:[10,2,1,""],std:[10,2,1,""],write:[10,2,1,""]},"sconce.monitors.stdout_monitor":{StdoutMonitor:[10,0,1,""]},"sconce.monitors.stdout_monitor.StdoutMonitor":{start_session:[10,2,1,""],write:[10,2,1,""]},"sconce.rate_controllers":{CompositeRateController:[5,0,1,""],ConstantRateController:[5,0,1,""],CosineRateController:[5,0,1,""],ExponentialRateController:[5,0,1,""],LinearRateController:[5,0,1,""],RateController:[5,0,1,""],StepRateController:[5,0,1,""],TriangleRateController:[5,0,1,""],base:[11,4,0,"-"],constant_rate_controller:[11,4,0,"-"],cosine_rate_controller:[11,4,0,"-"],exponential_rate_controller:[11,4,0,"-"],linear_rate_controller:[11,4,0,"-"],step_rate_controller:[11,4,0,"-"],triangle_rate_controller:[11,4,0,"-"]},"sconce.rate_controllers.RateController":{new_learning_rate:[5,2,1,""],start_session:[5,2,1,""]},"sconce.rate_controllers.base":{CompositeRateController:[11,0,1,""],RateController:[11,0,1,""]},"sconce.rate_controllers.base.CompositeRateController":{add_rate_controller:[11,2,1,""],new_learning_rate:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.base.RateController":{new_learning_rate:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.constant_rate_controller":{ConstantRateController:[11,0,1,""]},"sconce.rate_controllers.constant_rate_controller.ConstantRateController":{new_learning_rate:[11,2,1,""],reset_monitor:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.cosine_rate_controller":{CosineRateController:[11,0,1,""]},"sconce.rate_controllers.cosine_rate_controller.CosineRateController":{new_learning_rate:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.exponential_rate_controller":{ExponentialRateController:[11,0,1,""]},"sconce.rate_controllers.exponential_rate_controller.ExponentialRateController":{new_learning_rate:[11,2,1,""],should_continue:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.linear_rate_controller":{LinearRateController:[11,0,1,""]},"sconce.rate_controllers.linear_rate_controller.LinearRateController":{new_learning_rate:[11,2,1,""],should_continue:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.step_rate_controller":{StepRateController:[11,0,1,""]},"sconce.rate_controllers.step_rate_controller.StepRateController":{new_learning_rate:[11,2,1,""],start_session:[11,2,1,""]},"sconce.rate_controllers.triangle_rate_controller":{TriangleRateController:[11,0,1,""]},"sconce.rate_controllers.triangle_rate_controller.TriangleRateController":{new_learning_rate:[11,2,1,""],start_session:[11,2,1,""]},"sconce.trainer":{Trainer:[13,0,1,""]},"sconce.trainer.Trainer":{checkpoint:[13,2,1,""],load_model_state:[13,2,1,""],multi_train:[13,2,1,""],num_trainable_parameters:[13,1,1,""],restore:[13,2,1,""],save_model_state:[13,2,1,""],survey_learning_rate:[13,2,1,""],test:[13,2,1,""],train:[13,2,1,""]},"sconce.trainers":{AutoencoderTrainer:[13,0,1,""],ClassifierTrainer:[13,0,1,""],autoencoder_trainer:[12,4,0,"-"],classifier_trainer:[12,4,0,"-"]},"sconce.trainers.autoencoder_trainer":{AutoencoderMixin:[12,0,1,""],AutoencoderTrainer:[12,0,1,""]},"sconce.trainers.autoencoder_trainer.AutoencoderMixin":{plot_input_output_pairs:[12,2,1,""],plot_latent_space:[12,2,1,""]},"sconce.trainers.classifier_trainer":{ClassifierMixin:[12,0,1,""],ClassifierTrainer:[12,0,1,""]},"sconce.trainers.classifier_trainer.ClassifierMixin":{get_classification_accuracy:[12,2,1,""],get_confusion_matrix:[12,2,1,""],plot_confusion_matrix:[12,2,1,""],plot_samples:[12,2,1,""]},"sconce.utils":{Progbar:[6,0,1,""]},"sconce.utils.Progbar":{add:[6,2,1,""],update:[6,2,1,""]},sconce:{models:[2,4,0,"-"],trainer:[6,4,0,"-"],utils:[6,4,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","attribute","Python attribute"],"2":["py","method","Python method"],"3":["py","classmethod","Python class method"],"4":["py","module","Python module"],"5":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:attribute","2":"py:method","3":"py:classmethod","4":"py:module","5":"py:function"},terms:{"0x7fb1fbd498d0":[4,10],"100x1x28x28":[0,7],"6th":[6,13],"case":[4,10],"class":[0,1,2,4,5,6,7,8,9,10,11,12,13],"float":[0,4,5,6,7,10,11,13],"function":[0,2,5,7,11],"import":[4,5,10,11],"int":[0,2,4,5,6,7,8,10,11,13],"new":[0,2,5,7,8,11],"return":[0,2,5,6,7,11,13],"true":[0,2,4,7,8,9,10,12],For:[0,2,6,7,13],The:[0,2,4,5,6,7,10,11,13],These:[6,13],Using:[4,5,10,11],__next__:[0,7],_input:[4,10],_output:[4,10],_target:[4,10],abc:[10,11,12],abov:[2,8],accept:[2,4,10],access:[4,10],activ:[2,8,9],adaptiveaveragepooling2dlay:8,add:6,add_monitor:10,add_rate_control:11,added:[5,11],addit:[4,10],adjust:[5,6,11,13],after:[4,5,6,10,11,13],align:[5,11],all:[4,5,6,10,11,13],allow:[4,5,6,10,11,13],along:[6,13],also:[6,13],analyz:[6,13],ani:[0,2,7],anoth:[6,13],api_kei:[4,10],arbitrari:2,arg:[0,7,10,11],argument:[0,2,6,7,13],around:[0,7],autoencod:[2,8,12],autoencoder_train:[3,13],autoencodermixin:[12,13],autoencodertrain:[1,12],automat:[0,7],averag:6,back:[2,5,6,11,13],bar:6,barchart:[0,7],base:[0,2,3,4,5,6,8,9,12,13],basic:[1,2,8],basic_autoencod:3,basic_classifi:3,basicautoencod:[1,8],basicclassif:1,basicclassifi:[2,8],basicconvolutionalautoencod:1,batch:[6,13],batch_multipl:[6,13],batch_multipli:[4,5,6,10,11,13],batch_siz:[0,7],been:[4,10],befor:[0,5,6,7,11,13],begin:[0,5,7,11],behavior:[0,2,7],below:2,between:[5,11],binari:[2,8],blacklist:[4,10],bool:[0,7],both:[0,7],built:[2,8],cache_result:12,calcul:2,calculate_loss:[2,8],calculate_metr:[2,8],call:[4,5,6,10,11,13],callabl:[0,7],can:[2,4,5,6,8,10,11,13],capac:[4,10],channel:[0,2,7,8],checkpoint:[6,13],chosen:[6,13],cifar100:[0,7],cifar10:[0,7],classifi:[2,8],classification_accuraci:[2,8],classifier_train:[3,13],classifiermixin:[12,13],classifiertrain:[1,12],classmethod:[0,2,7,8,10],collect:[5,11],come:2,complet:[4,5,6,10,11,13],compos:[4,5,10,11],composit:[6,13],compositemonitor:[1,10],compositeratecontrol:[1,11],connect:[2,8],consist:[6,13],constant:[5,11],constant_rate_control:3,constantratecontrol:[1,11],constraint:2,construct:[2,8],constructor:[0,6,7,13],contain:[0,7],content:[2,8],control:[1,11],conv_channel:2,convolut:[2,8],convolution2d_lay:8,convolution2dlay:[2,8,9],convolutional_layer_attribut:[2,8],convolutional_layer_kwarg:[2,8],convolutional_layer_valu:[2,8],copi:[0,7],correctli:[2,8],cosin:[5,11],cosine_rate_control:3,cosineratecontrol:[1,6,11,13],cours:[5,11],cpu:[0,7],creat:[0,6,7,13],cross:[2,8],cuda:[0,7],current:6,cycl:[6,13],cycle_length:[6,13],cycle_multipli:[6,13],data:[1,4,5,6,7,10,11,13],data_gener:[0,3,6,12],data_load:[0,7],data_loc:[0,7],datafram:[0,7],dataframe_monitor:[3,4],dataframemonitor:[1,6,10,13],datagener:[1,6,7,13],dataload:[0,7],dataset:[0,2,6,7,8,13],dataset_class:[0,7],dataset_kwarg:[0,7],decod:[2,8],defin:[2,4,5,6,10,11,13],densli:[2,8],depend:[0,6,7,13],deprec:[0,7],depth:[2,8],describ:[2,5,8,11],detail:[0,2,7,8],detect:[5,11],determin:[2,6,8,13],devic:[0,7],dict:[0,2,4,5,7,8,10,11],dictionari:[0,2,7,8],differ:[2,8],directli:[0,7],directori:[0,7],displai:6,divis:[2,8],download:[0,7],drop:[5,11],drop_factor:[5,11],dropout:[2,8,9],dure:[4,5,6,10,11,13],each:[0,2,7,8],earli:[6,13],ect:[5,6,11,13],effect:[6,13],els:6,encod:[2,8],end:[4,5,10,11],end_sess:[4,10],endlessli:[0,7],entropi:[2,8],epoch:[0,6,7,13],equal:[6,13],evalu:[4,5,10,11],everi:[0,7],exampl:[0,2,6,7,8,13],expect:[2,4,5,6,10,11],exponenti:[5,11],exponential_rate_control:[3,6,13],exponentialratecontrol:[1,6,11,13],extern:[6,13],factor:[6,13],fall:[5,11],fals:[0,7,8,9],fashionmnist:[0,2,7,8],few:2,fig:10,figsiz:[10,12],figure_kwarg:10,figure_width:12,file:[2,6,8,13],filenam:[2,6,8,10,13],first:[0,6,7,13],floattensor:[0,7],folder:[0,7],follow:[2,5,8,11],form:2,forward:[2,8,9],fraction:[0,2,4,5,7,8,10,11],freeze_batchnorm_lay:8,from:[0,2,4,5,6,7,8,10,11,13],from_dataset:[0,7],from_fil:10,from_image_fold:[0,7],from_pytorch:[0,7],from_torchvis:[0,7],frome:[6,13],fulli:[2,8],fully_connected_lay:8,fully_connected_layer_attribut:[2,8],fully_connected_layer_kwarg:[2,8],fully_connected_layer_valu:[2,8],fullyconnectedlay:[2,8,9],futur:[4,10],gain:[4,10],gener:[1,7],get:2,get_classification_accuraci:12,get_confusion_matrix:12,get_image_info:7,get_summary_df:[0,7],given:[6,13],gpu:[0,7],gradient:2,greater:[6,13],group:[5,11],handi:[0,7],has:[4,5,6,10,11,13],have:[0,4,7,10],heatmap_kwarg:12,height:[2,8],hidden:[2,8],hidden_s:[2,8],histori:10,how:[0,2,6,7,8,13],imag:[0,2,7,8],image_channel:[2,8],image_data_gener:3,image_height:[2,8,12],image_width:[2,8],imagedatagener:[1,7],imagefold:[0,7],implement:[5,11],in_channel:[8,9],in_height:9,in_siz:9,in_width:9,includ:[2,4,5,10,11],increas:[6,13],index:[1,6],individu:[0,7],inf:[2,4,5,6,8,10,11,13],infer:[6,13],inplace_activ:[8,9],input:[0,2,6,7,8,13],instanti:[0,7],instead:[0,7],integ:[2,8],interfac:[4,5,10,11],interv:6,is_blacklist:10,iter:[0,4,5,6,7,10,11],its:[5,11],just:[2,4,5,10,11],kei:[0,2,4,5,7,10,11],kept:[6,13],kernel_s:[2,8,9],keyword:[0,2,6,7,13],kwarg:[0,4,6,7,8,10,13],label:[0,2,7,8,12],larg:[0,6,7,13],latent:[2,8,12],latent_s:[2,8],later:[6,13],layer:[2,8],layer_attribut:[2,8],layer_kwarg:[2,8],layer_valu:[2,8],learn:[4,5,6,10,11,13],learning_r:[4,5,10,11],least:2,len:[0,7],length:[2,6,13],like:[0,2,4,5,7,8,10,11],limit:[4,5,10,11],linear_rate_control:3,linearli:[5,11],linearratecontrol:[1,6,11,13],list:[2,6,8],live:[0,7],load:[0,7],load_model_st:[6,13],loader:[0,7],loader_kwarg:[0,7],locat:[0,6,7,13],longtensor:[0,7],look:[0,7],loop:[6,13],loss:[2,4,5,6,8,10,11,13],loss_kei:[5,11],losswise_monitor:[3,4],losswisemonitor:[1,10],made:2,mai:[2,4,5,10,11],make:[6,13],mani:[0,2,7,8],max_graph:[4,10],max_learning_r:[5,6,11,13],maximum:[6,13],mean:10,mediumseagreen:10,memori:[0,6,7,13],metadata:[0,4,7,10],method:[0,2,4,6,7,10,13],metric:[2,4,5,6,8,10,11],metric_nam:[4,10],min_graph:[4,10],min_learning_r:[5,6,11,13],minimum:[6,13],mnist:[0,2,6,7,8,13],mode:[6,13],model:[1,3,6,12,13],modifi:2,modul:[1,2,3,13],monitor:[1,3,6,12,13],more:[2,4,5,6,8,10,11,13],move:[5,11],movement_index:10,movement_kei:[5,11],movement_threshold:[5,11],movement_window:[5,11],much:[0,7],multi:[2,8],multi_train:[6,13],multilayer_perceptron:3,multilayerperceptron:[1,8],multipl:[5,6,11,13],multipli:[6,13],must:[2,4,5,10,11],name:[4,6,10],network:[2,8],never:[6,13],new_from_yaml_fil:[2,8],new_from_yaml_filenam:[2,8],new_learning_r:[5,11],next:[0,5,7,11],none:[0,2,4,5,6,7,8,10,11,12,13],note:2,num_block:8,num_categori:[2,8],num_channel:[0,7],num_col:12,num_cycl:[6,13],num_drop:[5,11],num_epoch:[6,13],num_sampl:[0,7,12],num_step:[4,5,10,11],num_trainable_paramet:[6,13],num_work:[0,7],number:[0,2,4,5,6,7,8,10,11,13],object:[2,4,6,7,8,10,13],occur:[6,13],often:[6,13],one:[6,13],onli:[5,6,11,13],oper:[4,10],optim:[2,5,6,11,12,13],option:[4,5,6,10,11,13],orchestr:[6,13],order:[5,11],ordereddict:[5,11],origin:[0,7],otehr:2,other:[2,4,6,10,11,13],otherwis:[0,7],out_channel:[2,8,9],out_height:9,out_siz:[2,8,9],out_width:9,output:[2,4,5,8,10,11],output_s:8,over:[5,6,11],packag:3,pad:[2,8,9],panda:[0,7],paper:[2,8],param:[4,10],paramet:[0,2,4,5,6,7,8,10,11,13],parameter_group:[5,11],pass:[0,2,4,5,6,7,10,11,13],path:[0,2,6,7,8,13],perceptron:[2,8],pil:[0,7],pin:[0,7],pin_memori:[0,7],pixel:[2,8],plot:[0,6,7,10,13],plot_confusion_matrix:12,plot_input_output_pair:12,plot_kwarg:10,plot_label_summari:[0,7],plot_latent_spac:12,plot_learning_rate_survei:10,plot_sampl:12,plot_size_summari:[0,7],posit:[2,8],possibl:[4,5,10,11],practic:[6,13],preactiv:[8,9],predict:[2,8,12],predicted_label:12,preprocess:7,previou:[6,13],previous:[6,13],produc:[4,10],progbar:6,progbar_kwarg:[4,10],progress:6,propag:[2,6,13],provid:[0,7],put:[0,7],python:2,pytorch:[0,6,7,13],rate:[1,4,6,10,11,13],rate_control:[3,5,6,12,13],rate_controller_class:[6,13],rate_controller_kwarg:[6,13],rate_monitor:[6,13],ratecontrol:[1,6,11,13],reach:[0,7],read:[2,8],real_dataset:[0,7],record:[4,6,10,13],refer:[0,1,7],rel:[6,13],relat:[2,8],remain:[2,5,8,11],remov:[0,7],represent:[2,8,12],reset:[0,7],reset_monitor:11,reshuffl:[0,7],resnet:[2,8],restor:[6,13],restrict:2,result:12,retain:[6,13],ringbuffer_monitor:[3,4],ringbuffermonitor:[1,10],rise:[5,6,11,12,13],root:[0,7],run:[6,13],sampl:[6,12,13],satisfi:2,save:[6,10,13],save_model_st:[6,13],scale:[5,11],scatter:[0,7],sconc:[0,2,4,5,13],score:12,screen:6,second:6,see:[0,2,4,5,6,7,8,10,11,13],self:[6,13],semi:6,sent:[6,13],session:[4,5,6,10,11,13],set:[0,7],shift:[5,11],should:[0,2,5,6,7,8,11],should_continu:11,show:[0,7],shuffl:[0,7],silent:6,simul:[6,13],singl:[4,5,10,11],size:[0,6,7,13],skip_first:10,smooth_window:10,some:[0,2,5,7,8,11],sort_bi:12,sourc:[0,2,4,5,6,7,8,9,10,11,12,13],specifi:[0,7],start:[0,2,4,5,7,10,11],start_sess:[4,5,10,11],state:[6,13],stateful_metr:6,std:10,stdout_monitor:[3,4],stdoutmonitor:[1,6,10,13],step:[4,5,6,10,11],step_rate_control:3,stepratecontrol:[1,11],stop:[5,6,11,13],stop_factor:[5,6,11,13],store:[0,7],str:[4,10],stride:[2,8,9],string:6,subclass:[2,6,13],submodul:3,subpackag:3,subprocess:[0,7],subset:[0,7],summar:[0,7],support:[0,7],survei:[6,13],survey_learning_r:[6,13],system:[0,6,7,13],tag:[4,10],take:[0,7],target:[0,2,6,7,8,13],task:[4,10],temporari:[0,6,7,13],tensor:[0,2,7],test:[0,4,6,7,10,13],test_color:10,test_data_gener:[6,12,13],test_loss:[4,5,10,11],test_to_train_ratio:[6,13],than:[6,13],them:[0,7],thi:[0,2,4,5,6,7,8,10,11,13],thin:[0,7],three:[2,8],through:[0,6,7,13],time:[5,6,11],titl:[10,12],togeth:[4,5,10,11],tomato:10,torch:[0,2,6,7,8,9,13],torchvis:[0,7],total:[2,6,8],totensor:[0,7],train:[0,4,5,6,7,10,11,13],trainabl:[6,13],trainer:[1,2,3,4,5,10,11],training_color:10,training_data_gener:[6,12,13],training_loss:[4,5,10,11],transform:[0,7],triangle_rate_control:3,triangleratecontrol:[1,11],true_label:12,tupl:[0,6,7],two:[0,4,5,7,10,11],type:[0,5,6,7,11,13],typic:[6,13],underli:[0,6,7,13],unfreeze_batchnorm_lay:8,unknown:6,until:[6,13],updat:[4,5,6,10,11,13],usag:[6,13],use:[0,4,6,7,10,13],used:[0,2,4,5,6,7,10,11,13],uses:[2,4,8,10],using:[4,10],util:3,val_loss:[4,10],valu:[0,2,5,6,7,8,11,13],value_for_last_step:6,variou:[6,13],verbos:6,version:[0,7],visual:6,want:[4,5,10,11],were:[6,13],what:[5,11],when:[4,5,10,11],where:[0,2,5,6,7,8,11,13],which:[0,7],wide:[2,8],wide_resnet_image_classifi:3,widening_factor:[2,8],wideresnetblock_3x3:8,wideresnetgroup_3x3:8,wideresnetimageclassifi:[1,8],width:[2,6,8],with_batchnorm:[2,8,9],without:[6,13],would:[6,13],wrap:[0,7],wrapper:[0,7],write:[4,10],x_in:[8,9],x_latent:8,yaml:[2,8],yaml_fil:[2,8],yaml_filenam:[2,8],yield:[0,6,7,13],you:[2,4,5,6,10,11,13]},titles:["Data Generators","Sconce Documentation","Models","sconce","Monitors","Rate Controllers","sconce package","sconce.data_generators package","sconce.models package","sconce.models.layers package","sconce.monitors package","sconce.rate_controllers package","sconce.trainers package","Trainers"],titleterms:{autoencoder_train:12,autoencodertrain:13,base:[7,10,11],basic:13,basic_autoencod:8,basic_classifi:8,basicautoencod:2,basicclassif:2,basicconvolutionalautoencod:2,classifier_train:12,classifiertrain:13,compositemonitor:4,compositeratecontrol:5,constant_rate_control:11,constantratecontrol:5,control:5,convolution2d_lay:9,cosine_rate_control:11,cosineratecontrol:5,data:0,data_gener:7,dataframe_monitor:10,dataframemonitor:4,datagener:0,document:1,exponential_rate_control:11,exponentialratecontrol:5,fully_connected_lay:9,gener:0,image_data_gener:7,imagedatagener:0,indic:1,layer:9,linear_rate_control:11,linearratecontrol:5,losswise_monitor:10,losswisemonitor:4,model:[2,8,9],modul:[6,7,8,9,10,11,12],monitor:[4,10],multilayer_perceptron:8,multilayerperceptron:2,packag:[6,7,8,9,10,11,12],rate:5,rate_control:11,ratecontrol:5,ringbuffer_monitor:10,ringbuffermonitor:4,sconc:[1,3,6,7,8,9,10,11,12],stdout_monitor:10,stdoutmonitor:4,step_rate_control:11,stepratecontrol:5,submodul:[6,8],subpackag:[6,8],tabl:1,trainer:[6,12,13],triangle_rate_control:11,triangleratecontrol:5,util:6,wide_resnet_image_classifi:8,wideresnetimageclassifi:2}})