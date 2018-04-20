Search.setIndex({docnames:["data_generator","index","models","modules","monitors","rate_controllers","sconce","sconce.models","sconce.models.layers","sconce.monitors","sconce.rate_controllers","sconce.trainers","trainers"],envversion:53,filenames:["data_generator.rst","index.rst","models.rst","modules.rst","monitors.rst","rate_controllers.rst","sconce.rst","sconce.models.rst","sconce.models.layers.rst","sconce.monitors.rst","sconce.rate_controllers.rst","sconce.trainers.rst","trainers.rst"],objects:{"sconce.data_generator":{DataGenerator:[6,1,1,""]},"sconce.data_generator.DataGenerator":{batch_size:[6,2,1,""],cuda:[6,3,1,""],dataset:[6,2,1,""],from_dataset:[6,4,1,""],from_pytorch:[6,4,1,""],next:[6,3,1,""],num_samples:[6,2,1,""],preprocess:[6,3,1,""],reset:[6,3,1,""]},"sconce.models":{BasicAutoencoder:[2,1,1,""],BasicClassifier:[2,1,1,""],MultilayerPerceptron:[2,1,1,""],WideResnetImageClassifier:[2,1,1,""],basic_autoencoder:[7,0,0,"-"],basic_classifier:[7,0,0,"-"],multilayer_perceptron:[7,0,0,"-"],wide_resnet_image_classifier:[7,0,0,"-"]},"sconce.models.BasicClassifier":{new_from_yaml_file:[2,4,1,""],new_from_yaml_filename:[2,4,1,""]},"sconce.models.MultilayerPerceptron":{new_from_yaml_file:[2,4,1,""],new_from_yaml_filename:[2,4,1,""]},"sconce.models.basic_autoencoder":{BasicAutoencoder:[7,1,1,""]},"sconce.models.basic_autoencoder.BasicAutoencoder":{calculate_loss:[7,3,1,""],decode:[7,3,1,""],encode:[7,3,1,""],forward:[7,3,1,""]},"sconce.models.basic_classifier":{BasicClassifier:[7,1,1,""]},"sconce.models.basic_classifier.BasicClassifier":{calculate_loss:[7,3,1,""],calculate_metrics:[7,3,1,""],forward:[7,3,1,""],freeze_batchnorm_layers:[7,3,1,""],layers:[7,2,1,""],new_from_yaml_file:[7,4,1,""],new_from_yaml_filename:[7,4,1,""],unfreeze_batchnorm_layers:[7,3,1,""]},"sconce.models.layers":{convolution2d_layer:[8,0,0,"-"],fully_connected_layer:[8,0,0,"-"]},"sconce.models.layers.convolution2d_layer":{Convolution2dLayer:[8,1,1,""]},"sconce.models.layers.convolution2d_layer.Convolution2dLayer":{forward:[8,3,1,""],out_height:[8,3,1,""],out_width:[8,3,1,""]},"sconce.models.layers.fully_connected_layer":{FullyConnectedLayer:[8,1,1,""]},"sconce.models.layers.fully_connected_layer.FullyConnectedLayer":{forward:[8,3,1,""]},"sconce.models.multilayer_perceptron":{MultilayerPerceptron:[7,1,1,""]},"sconce.models.multilayer_perceptron.MultilayerPerceptron":{calculate_loss:[7,3,1,""],calculate_metrics:[7,3,1,""],forward:[7,3,1,""],new_from_yaml_file:[7,4,1,""],new_from_yaml_filename:[7,4,1,""]},"sconce.models.wide_resnet_image_classifier":{AdaptiveAveragePooling2dLayer:[7,1,1,""],WideResnetBlock_3x3:[7,1,1,""],WideResnetGroup_3x3:[7,1,1,""],WideResnetImageClassifier:[7,1,1,""]},"sconce.models.wide_resnet_image_classifier.AdaptiveAveragePooling2dLayer":{forward:[7,3,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetBlock_3x3":{forward:[7,3,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetGroup_3x3":{forward:[7,3,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetImageClassifier":{calculate_loss:[7,3,1,""],calculate_metrics:[7,3,1,""],forward:[7,3,1,""]},"sconce.monitors":{CompositeMonitor:[4,1,1,""],DataframeMonitor:[4,1,1,""],LosswiseMonitor:[4,1,1,""],Monitor:[4,1,1,""],RingbufferMonitor:[4,1,1,""],StdoutMonitor:[4,1,1,""],base:[9,0,0,"-"],dataframe_monitor:[9,0,0,"-"],losswise_monitor:[9,0,0,"-"],ringbuffer_monitor:[9,0,0,"-"],stdout_monitor:[9,0,0,"-"]},"sconce.monitors.Monitor":{end_session:[4,3,1,""],start_session:[4,3,1,""],write:[4,3,1,""]},"sconce.monitors.base":{CompositeMonitor:[9,1,1,""],Monitor:[9,1,1,""]},"sconce.monitors.base.CompositeMonitor":{add_monitor:[9,3,1,""],end_session:[9,3,1,""],start_session:[9,3,1,""],write:[9,3,1,""]},"sconce.monitors.base.Monitor":{end_session:[9,3,1,""],start_session:[9,3,1,""],write:[9,3,1,""]},"sconce.monitors.dataframe_monitor":{DataframeMonitor:[9,1,1,""]},"sconce.monitors.dataframe_monitor.DataframeMonitor":{df:[9,2,1,""],from_file:[9,4,1,""],is_blacklisted:[9,3,1,""],plot:[9,3,1,""],plot_learning_rate_survey:[9,3,1,""],save:[9,3,1,""],start_session:[9,3,1,""],write:[9,3,1,""]},"sconce.monitors.losswise_monitor":{LosswiseMonitor:[9,1,1,""]},"sconce.monitors.losswise_monitor.LosswiseMonitor":{start_session:[9,3,1,""],write:[9,3,1,""]},"sconce.monitors.ringbuffer_monitor":{RingbufferMonitor:[9,1,1,""]},"sconce.monitors.ringbuffer_monitor.RingbufferMonitor":{mean:[9,3,1,""],movement_index:[9,2,1,""],start_session:[9,3,1,""],std:[9,3,1,""],write:[9,3,1,""]},"sconce.monitors.stdout_monitor":{StdoutMonitor:[9,1,1,""]},"sconce.monitors.stdout_monitor.StdoutMonitor":{start_session:[9,3,1,""],write:[9,3,1,""]},"sconce.rate_controllers":{ConstantRateController:[5,1,1,""],CosineRateController:[5,1,1,""],ExponentialRateController:[5,1,1,""],LinearRateController:[5,1,1,""],StepRateController:[5,1,1,""],TriangleRateController:[5,1,1,""],base:[10,0,0,"-"],constant_rate_controller:[10,0,0,"-"],cosine_rate_controller:[10,0,0,"-"],exponential_rate_controller:[10,0,0,"-"],linear_rate_controller:[10,0,0,"-"],step_rate_controller:[10,0,0,"-"],triangle_rate_controller:[10,0,0,"-"]},"sconce.rate_controllers.base":{RateController:[10,1,1,""]},"sconce.rate_controllers.base.RateController":{new_learning_rate:[10,3,1,""],start_session:[10,3,1,""]},"sconce.rate_controllers.constant_rate_controller":{ConstantRateController:[10,1,1,""]},"sconce.rate_controllers.constant_rate_controller.ConstantRateController":{new_learning_rate:[10,3,1,""],reset_monitor:[10,3,1,""],start_session:[10,3,1,""]},"sconce.rate_controllers.cosine_rate_controller":{CosineRateController:[10,1,1,""]},"sconce.rate_controllers.cosine_rate_controller.CosineRateController":{new_learning_rate:[10,3,1,""],start_session:[10,3,1,""]},"sconce.rate_controllers.exponential_rate_controller":{ExponentialRateController:[10,1,1,""]},"sconce.rate_controllers.exponential_rate_controller.ExponentialRateController":{new_learning_rate:[10,3,1,""],should_continue:[10,3,1,""],start_session:[10,3,1,""]},"sconce.rate_controllers.linear_rate_controller":{LinearRateController:[10,1,1,""]},"sconce.rate_controllers.linear_rate_controller.LinearRateController":{new_learning_rate:[10,3,1,""],should_continue:[10,3,1,""],start_session:[10,3,1,""]},"sconce.rate_controllers.step_rate_controller":{StepRateController:[10,1,1,""]},"sconce.rate_controllers.step_rate_controller.StepRateController":{new_learning_rate:[10,3,1,""],start_session:[10,3,1,""]},"sconce.rate_controllers.triangle_rate_controller":{TriangleRateController:[10,1,1,""]},"sconce.rate_controllers.triangle_rate_controller.TriangleRateController":{new_learning_rate:[10,3,1,""],start_session:[10,3,1,""]},"sconce.trainer":{Trainer:[12,1,1,""]},"sconce.trainer.Trainer":{checkpoint:[12,3,1,""],load_model_state:[12,3,1,""],multi_train:[12,3,1,""],num_trainable_parameters:[12,2,1,""],restore:[12,3,1,""],save_model_state:[12,3,1,""],survey_learning_rate:[12,3,1,""],test:[12,3,1,""],train:[12,3,1,""]},"sconce.trainers":{AutoencoderTrainer:[12,1,1,""],ClassifierTrainer:[12,1,1,""],autoencoder_trainer:[11,0,0,"-"],classifier_trainer:[11,0,0,"-"]},"sconce.trainers.autoencoder_trainer":{AutoencoderMixin:[11,1,1,""],AutoencoderTrainer:[11,1,1,""]},"sconce.trainers.autoencoder_trainer.AutoencoderMixin":{plot_input_output_pairs:[11,3,1,""],plot_latent_space:[11,3,1,""]},"sconce.trainers.classifier_trainer":{ClassifierMixin:[11,1,1,""],ClassifierTrainer:[11,1,1,""]},"sconce.trainers.classifier_trainer.ClassifierMixin":{get_classification_accuracy:[11,3,1,""],get_confusion_matrix:[11,3,1,""],plot_confusion_matrix:[11,3,1,""],plot_samples:[11,3,1,""]},"sconce.utils":{Progbar:[6,1,1,""]},"sconce.utils.Progbar":{add:[6,3,1,""],update:[6,3,1,""]},sconce:{data_generator:[6,0,0,"-"],trainer:[6,0,0,"-"],utils:[6,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:classmethod"},terms:{"0x7fb1fbd498d0":[4,9],"100x1x28x28":[0,6],"6th":[6,12],"case":[4,9],"class":[0,1,2,4,5,6,7,8,9,10,11,12],"float":[0,4,6,9,12],"function":[0,5,6,10],"import":[4,9],"int":[0,2,4,6,7,9,12],"new":[2,7],"return":[0,6,12],"true":[0,2,4,6,7,8,9,11],For:[0,6,12],The:[4,5,6,9,10,12],These:[6,12],Using:[4,9],__next__:[0,6],_input:[4,9],_output:[4,9],abc:[9,10,11],abov:[2,7],accept:[4,9],access:[4,9],activ:[2,7,8],adaptiveaveragepooling2dlay:7,add:6,add_monitor:9,addit:[4,9],adjust:[5,6,10,12],after:[4,5,6,9,10,12],all:[4,6,9,12],allow:[4,6,9,12],along:[6,12],also:[6,12],analyz:[6,12],anoth:[6,12],api_kei:[4,9],arg:[9,10],argument:[0,6,12],around:[0,6],autoencod:[2,7,11],autoencoder_train:[3,12],autoencodermixin:[11,12],autoencodertrain:[1,11],automat:[0,6],averag:6,back:[5,6,10,12],bar:6,base:[2,3,4,5,6,7,8,11,12],basic:[1,2,7],basic_autoencod:3,basic_classifi:3,basicautoencod:[1,7],basicclassif:1,basicclassifi:[2,7],batch:[6,12],batch_multipl:[6,12],batch_multipli:[4,6,9,12],batch_siz:[0,6],been:[4,9],befor:[0,6,12],begin:[0,5,6,10],behavior:[0,6],between:[5,10],binari:[2,7],blacklist:[4,9],bool:[0,6],both:[0,6],built:[2,7],cache_result:11,calculate_loss:7,calculate_metr:7,call:[4,6,9,12],callabl:[0,6],can:[2,4,5,6,7,9,10,12],capac:[4,9],channel:[2,7],checkpoint:[6,12],chosen:[6,12],cifar100:[0,6],cifar10:[0,6],classifi:[2,7],classification_accuraci:[2,7],classifier_train:[3,12],classifiermixin:[11,12],classifiertrain:[1,11],classmethod:[0,2,6,7,9],complet:[4,6,9,12],compos:[4,9],composit:[6,12],compositemonitor:[1,9],connect:[2,7],consist:[6,12],constant:[5,10],constant_rate_control:3,constantratecontrol:[1,10],construct:[2,7],constructor:[0,6,12],contain:[0,6],content:[2,7],control:1,convolut:[2,7],convolution2d_lay:7,convolution2dlay:[2,7,8],convolutional_layer_attribut:[2,7],convolutional_layer_kwarg:[2,7],convolutional_layer_valu:[2,7],copi:[0,6],correctli:[2,7],cosin:[5,10],cosine_rate_control:3,cosineratecontrol:[1,6,10,12],cours:[5,10],cpu:[0,6],creat:[0,6,12],cross:[2,7],cuda:[0,6],current:6,cycl:[6,12],cycle_length:[6,12],cycle_multipli:[6,12],data:[0,4,6,9,10,12],data_gener:[0,3,11],data_load:[0,6],data_loc:[0,6],dataframe_monitor:[3,4],dataframemonitor:[1,6,9,12],datagener:[1,6,12],dataload:[0,6],dataset:[0,2,6,7,12],dataset_class:[0,6],decod:[2,7],defin:[4,6,9,12],densli:[2,7],depend:[0,6,12],depth:[2,7],describ:[2,7],detail:[0,2,6,7],detect:[5,10],determin:[2,6,7,12],devic:[0,6],dict:[0,2,4,6,7,9],dictionari:[0,2,6,7],differ:[2,7],directli:[0,6],displai:6,divis:[2,7],dodgerblu:9,download:[0,6],drop:[5,10],drop_factor:[5,10],dropout:[2,7,8],dure:[4,6,9,12],each:[2,7],earli:[6,12],ect:[6,12],effect:[6,12],els:6,encod:[2,7],end:[4,5,9,10],end_sess:[4,9],endlessli:[0,6],entropi:[2,7],epoch:[0,6,12],equal:[6,12],evalu:[4,9],everi:[0,6],exampl:[0,2,6,7,12],expect:[4,6,9],exponenti:[5,10],exponential_rate_control:[3,6,12],exponentialratecontrol:[1,6,10,12],extern:[6,12],factor:[6,12],fall:[5,10],fals:[0,6,7,8],fashionmnist:[0,2,6,7],fig:9,figsiz:[9,11],figure_kwarg:9,figure_width:11,file:[2,6,7,12],filenam:[2,6,7,9,12],first:[6,12],floattensor:[0,6],follow:[2,5,7,10],forward:[7,8],fraction:[0,2,4,6,7,9],freeze_batchnorm_lay:7,from:[0,2,4,5,6,7,9,10,12],from_dataset:[0,6],from_fil:9,from_pytorch:[0,6],frome:[6,12],fulli:[2,7],fully_connected_lay:7,fully_connected_layer_attribut:[2,7],fully_connected_layer_kwarg:[2,7],fully_connected_layer_valu:[2,7],fullyconnectedlay:[2,7,8],futur:[4,9],gain:[4,9],get_classification_accuraci:11,get_confusion_matrix:11,given:[6,12],gpu:[0,6],greater:[6,12],has:[4,5,6,9,10,12],have:[0,4,6,9],heatmap_kwarg:11,height:[2,7],hidden:[2,7],hidden_s:[2,7],histori:9,how:[0,2,6,7,12],imag:[0,2,6,7],image_channel:[2,7],image_height:[2,7,11],image_width:[2,7],in_channel:[7,8],in_height:8,in_siz:8,in_width:8,includ:[4,9],increas:[6,12],index:[1,6],individu:[0,6],inf:[2,4,6,7,9,12],infer:[6,12],inplace_activ:[7,8],input:[0,2,6,7,12],instanti:[0,6],instead:[0,6],integ:[2,7],interfac:[4,9],interv:6,is_blacklist:9,iter:[0,4,6,9],its:[5,10],just:[4,9],kei:[0,4,6,9],kept:[6,12],kernel_s:[2,7,8],keyword:[6,12],kwarg:[0,4,6,7,9,12],label:[2,7,11],larg:[0,6,12],latent:[2,7,11],latent_s:[2,7],later:[6,12],layer:[2,7],layer_attribut:[2,7],layer_kwarg:[2,7],layer_valu:[2,7],learn:[4,5,6,9,10,12],learning_r:[4,5,9,10],learning_rate_color:9,len:[0,6],length:[6,12],like:[0,2,4,6,7,9],limit:[4,9],linear_rate_control:3,linearli:[5,10],linearratecontrol:[1,6,10,12],list:[2,6,7],live:[0,6],load:[0,6],load_model_st:[6,12],loader:[0,6],locat:[0,6,12],longtensor:[0,6],loop:[6,12],loss:[2,4,5,6,7,9,10,12],loss_kei:[5,10],losswise_monitor:[3,4],losswisemonitor:[1,9],mai:[4,9],make:[6,12],mani:[0,2,6,7],max_graph:[4,9],max_learning_r:[5,6,10,12],maximum:[6,12],mean:9,mediumseagreen:9,memori:[0,6,12],metadata:[4,9],method:[0,4,6,9,12],metric:[2,4,5,6,7,9,10],metric_nam:[4,9],min_graph:[4,9],min_learning_r:[5,6,10,12],minimum:[6,12],mnist:[0,2,6,7,12],mode:[6,12],model:[1,3,6,11,12],modul:[1,3,12],monitor:[1,3,6,11,12],more:[2,4,6,7,9,12],move:[5,10],movement_index:9,movement_kei:[5,10],movement_threshold:[5,10],movement_window:[5,10],much:[0,6],multi:[2,7],multi_train:[6,12],multilayer_perceptron:3,multilayerperceptron:[1,7],multipl:[6,12],multipli:[6,12],must:[4,9],name:[4,6,9],network:[2,7],never:[6,12],new_from_yaml_fil:[2,7],new_from_yaml_filenam:[2,7],new_learning_r:10,next:[0,6],none:[0,2,4,5,6,7,9,10,11,12],num_block:7,num_categori:[2,7],num_col:11,num_cycl:[6,12],num_drop:[5,10],num_epoch:[6,12],num_sampl:[0,6,11],num_step:[4,5,9,10],num_trainable_paramet:[6,12],num_work:[0,6],number:[2,4,6,7,9,12],object:[0,2,4,6,7,9,12],occur:[6,12],often:[6,12],one:[6,12],onli:[6,12],oper:[4,9],optim:[6,11,12],option:[4,6,9,12],orchestr:[6,12],origin:[0,6],other:[4,6,9,12],otherwis:[0,6],out_channel:[2,7,8],out_height:8,out_siz:[2,7,8],out_width:8,output:[2,4,7,9],output_s:7,over:[5,6,10],packag:3,pad:[2,7,8],paper:[2,7],param:[4,9],paramet:[0,2,4,6,7,9,12],pass:[0,4,6,9,12],path:[0,2,6,7,12],perceptron:[2,7],pil:[0,6],pin:[0,6],pin_memori:[0,6],pixel:[2,7],plot:[6,9,12],plot_confusion_matrix:11,plot_input_output_pair:11,plot_kwarg:9,plot_latent_spac:11,plot_learning_rate_survei:9,plot_sampl:11,posit:[2,7],possibl:[4,9],practic:[6,12],preactiv:[7,8],predict:[2,7,11],predicted_label:11,preprocess:6,previou:[6,12],previous:[6,12],produc:[4,9],progbar:6,progbar_kwarg:[4,9],progress:6,propag:[6,12],put:[0,6],pytorch:[0,6,12],rate:[1,4,6,9,10,12],rate_control:[3,5,6,11,12],rate_controller_class:[6,12],rate_controller_kwarg:[6,12],rate_monitor:[6,12],ratecontrol:[6,10,12],read:[2,7],record:[4,6,9,12],refer:[0,1,6],rel:[6,12],relat:[2,7],remain:[2,5,7,10],represent:[2,7,11],reset:[0,6],reset_monitor:10,reshuffl:[0,6],resnet:[2,7],restor:[6,12],result:11,retain:[6,12],ringbuffer_monitor:[3,4],ringbuffermonitor:[1,9],rise:[5,6,10,11,12],root:[0,6],run:[6,12],sampl:[6,11,12],save:[6,9,12],save_model_st:[6,12],scale:[5,10],sconc:[0,2,4,5,12],score:11,screen:6,second:6,see:[0,2,4,6,7,9,12],self:[6,12],semi:6,sent:[6,12],session:[4,6,9,12],set:[0,6],shift:[5,10],should:[0,2,6,7],should_continu:10,shuffl:[0,6],silent:6,simul:[6,12],singl:[4,9],size:[0,6,12],skip_first:9,smooth_window:9,some:[2,5,7,10],sort_bi:11,sourc:[0,2,4,5,6,7,8,9,10,11,12],specifi:[0,6],start:[0,4,6,9],start_sess:[4,9,10],state:[6,12],stateful_metr:6,std:9,stdout_monitor:[3,4],stdoutmonitor:[1,6,9,12],step:[4,6,9,10],step_rate_control:3,stepratecontrol:[1,10],stop:[5,6,10,12],stop_factor:[5,6,10,12],store:[0,6],str:[4,9],stride:[2,7,8],string:6,subclass:[6,12],submodul:3,subpackag:3,subprocess:[0,6],support:[0,6],survei:[6,12],survey_learning_r:[6,12],system:[0,6,12],tag:[4,9],take:[0,6],target:[0,6,7,12],task:[4,9],temporari:[0,6,12],tensor:[0,6],test:[0,4,6,9,12],test_color:9,test_data_gener:[6,11,12],test_loss:[4,9],test_to_train_ratio:[6,12],than:[6,12],them:[0,6],thi:[0,2,4,6,7,9,12],thin:[0,6],through:[0,6,12],time:[5,6,10],titl:[9,11],togeth:[4,9],tomato:9,torch:[0,6,7,8,12],torchvis:[0,6],total:[2,6,7],totensor:[0,6],train:[0,4,6,9,12],trainabl:[6,12],trainer:[1,3,4,9],training_color:9,training_data_gener:[6,11,12],training_loss:[4,5,9,10],transform:[0,6],triangle_rate_control:3,triangleratecontrol:[1,10],true_label:11,tupl:[0,6],two:[0,2,4,6,7,9],type:[6,12],typic:[6,12],underli:[0,6,12],unfreeze_batchnorm_lay:7,unknown:6,until:[6,12],updat:[4,6,9,12],usag:[6,12],use:[0,4,6,9,12],used:[0,4,6,9,12],uses:[2,4,7,9],using:[4,9],util:3,val_loss:[4,9],valu:[0,2,6,7,12],value_for_last_step:6,variabl:[0,6],variou:[6,12],verbos:6,version:[0,6],visual:6,want:[4,9],were:[6,12],when:[4,9],where:[0,2,6,7,12],which:[0,6],wide:[2,7],wide_resnet_image_classifi:3,widening_factor:[2,7],wideresnetblock_3x3:7,wideresnetgroup_3x3:7,wideresnetimageclassifi:[1,7],width:[2,6,7],with_batchnorm:[2,7,8],without:[6,12],would:[6,12],wrap:[0,6],wrapper:[0,6],write:[4,9],x_in:[7,8],x_latent:7,yaml:[2,7],yaml_fil:[2,7],yaml_filenam:[2,7],yield:[0,6,12],you:[4,6,9,12]},titles:["<span class=\"hidden-section\">DataGenerator</span>","Sconce Documentation","Models","sconce","Monitors","Rate Controllers","sconce package","sconce.models package","sconce.models.layers package","sconce.monitors package","sconce.rate_controllers package","sconce.trainers package","Trainers"],titleterms:{autoencoder_train:11,autoencodertrain:12,base:[9,10],basic:12,basic_autoencod:7,basic_classifi:7,basicautoencod:2,basicclassif:2,classifier_train:11,classifiertrain:12,compositemonitor:4,constant_rate_control:10,constantratecontrol:5,control:5,convolution2d_lay:8,cosine_rate_control:10,cosineratecontrol:5,data_gener:6,dataframe_monitor:9,dataframemonitor:4,datagener:0,document:1,exponential_rate_control:10,exponentialratecontrol:5,fully_connected_lay:8,indic:1,layer:8,linear_rate_control:10,linearratecontrol:5,losswise_monitor:9,losswisemonitor:4,model:[2,7,8],modul:[6,7,8,9,10,11],monitor:[4,9],multilayer_perceptron:7,multilayerperceptron:2,packag:[6,7,8,9,10,11],rate:5,rate_control:10,ringbuffer_monitor:9,ringbuffermonitor:4,sconc:[1,3,6,7,8,9,10,11],stdout_monitor:9,stdoutmonitor:4,step_rate_control:10,stepratecontrol:5,submodul:[6,7],subpackag:[6,7],tabl:1,trainer:[6,11,12],triangle_rate_control:10,triangleratecontrol:5,util:6,wide_resnet_image_classifi:7,wideresnetimageclassifi:2}})