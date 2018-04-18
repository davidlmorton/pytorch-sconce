Search.setIndex({docnames:["index","modules","sconce","sconce.models","sconce.models.layers","sconce.monitors","sconce.rate_controllers","sconce.trainers"],envversion:53,filenames:["index.rst","modules.rst","sconce.rst","sconce.models.rst","sconce.models.layers.rst","sconce.monitors.rst","sconce.rate_controllers.rst","sconce.trainers.rst"],objects:{"":{sconce:[2,0,0,"-"]},"sconce.data_generator":{DataGenerator:[2,1,1,""]},"sconce.data_generator.DataGenerator":{batch_size:[2,2,1,""],cuda:[2,3,1,""],dataset:[2,2,1,""],from_dataset:[2,4,1,""],from_pytorch:[2,4,1,""],num_samples:[2,2,1,""],reset:[2,3,1,""]},"sconce.models":{basic_autoencoder:[3,0,0,"-"],basic_classifier:[3,0,0,"-"],convolutional_autoencoder_plus_perceptron:[3,0,0,"-"],layers:[4,0,0,"-"],multilayer_perceptron:[3,0,0,"-"],simple_convolutional_autoencoder:[3,0,0,"-"],wide_resnet_image_classifier:[3,0,0,"-"]},"sconce.models.basic_autoencoder":{BasicAutoencoder:[3,1,1,""]},"sconce.models.basic_autoencoder.BasicAutoencoder":{calculate_loss:[3,3,1,""],decode:[3,3,1,""],encode:[3,3,1,""],forward:[3,3,1,""]},"sconce.models.basic_classifier":{BasicClassifier:[3,1,1,""]},"sconce.models.basic_classifier.BasicClassifier":{calculate_loss:[3,3,1,""],calculate_metrics:[3,3,1,""],forward:[3,3,1,""],freeze_batchnorm_layers:[3,3,1,""],layers:[3,2,1,""],new_from_yaml_file:[3,4,1,""],new_from_yaml_filename:[3,4,1,""],unfreeze_batchnorm_layers:[3,3,1,""]},"sconce.models.convolutional_autoencoder_plus_perceptron":{ConvolutionalAutoencoderPlusPerceptron:[3,1,1,""],ConvolutionalLayer:[3,1,1,""],DeconvolutionalLayer:[3,1,1,""],Perceptron:[3,1,1,""]},"sconce.models.convolutional_autoencoder_plus_perceptron.ConvolutionalAutoencoderPlusPerceptron":{calculate_losses:[3,3,1,""],decode:[3,3,1,""],encode:[3,3,1,""],forward:[3,3,1,""]},"sconce.models.convolutional_autoencoder_plus_perceptron.ConvolutionalLayer":{forward:[3,3,1,""]},"sconce.models.convolutional_autoencoder_plus_perceptron.DeconvolutionalLayer":{forward:[3,3,1,""]},"sconce.models.convolutional_autoencoder_plus_perceptron.Perceptron":{forward:[3,3,1,""]},"sconce.models.layers":{convolution2d_layer:[4,0,0,"-"],fully_connected_layer:[4,0,0,"-"]},"sconce.models.layers.convolution2d_layer":{Convolution2dLayer:[4,1,1,""]},"sconce.models.layers.convolution2d_layer.Convolution2dLayer":{forward:[4,3,1,""],out_height:[4,3,1,""],out_width:[4,3,1,""]},"sconce.models.layers.fully_connected_layer":{FullyConnectedLayer:[4,1,1,""]},"sconce.models.layers.fully_connected_layer.FullyConnectedLayer":{forward:[4,3,1,""]},"sconce.models.multilayer_perceptron":{MultilayerPerceptron:[3,1,1,""]},"sconce.models.multilayer_perceptron.MultilayerPerceptron":{calculate_loss:[3,3,1,""],calculate_metrics:[3,3,1,""],forward:[3,3,1,""],new_from_yaml_file:[3,4,1,""],new_from_yaml_filename:[3,4,1,""]},"sconce.models.simple_convolutional_autoencoder":{ConvolutionalLayer:[3,1,1,""],DeconvolutionalLayer:[3,1,1,""],SimpleConvolutionalAutoencoder:[3,1,1,""]},"sconce.models.simple_convolutional_autoencoder.ConvolutionalLayer":{forward:[3,3,1,""]},"sconce.models.simple_convolutional_autoencoder.DeconvolutionalLayer":{forward:[3,3,1,""]},"sconce.models.simple_convolutional_autoencoder.SimpleConvolutionalAutoencoder":{calculate_loss:[3,3,1,""],decode:[3,3,1,""],encode:[3,3,1,""],forward:[3,3,1,""]},"sconce.models.wide_resnet_image_classifier":{AdaptiveAveragePooling2dLayer:[3,1,1,""],WideResnetBlock_3x3:[3,1,1,""],WideResnetGroup_3x3:[3,1,1,""],WideResnetImageClassifier:[3,1,1,""]},"sconce.models.wide_resnet_image_classifier.AdaptiveAveragePooling2dLayer":{forward:[3,3,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetBlock_3x3":{forward:[3,3,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetGroup_3x3":{forward:[3,3,1,""]},"sconce.models.wide_resnet_image_classifier.WideResnetImageClassifier":{calculate_loss:[3,3,1,""],calculate_metrics:[3,3,1,""],forward:[3,3,1,""]},"sconce.monitors":{base:[5,0,0,"-"],dataframe_monitor:[5,0,0,"-"],losswise_monitor:[5,0,0,"-"],ringbuffer_monitor:[5,0,0,"-"],stdout_monitor:[5,0,0,"-"]},"sconce.monitors.base":{CompositeMonitor:[5,1,1,""],Monitor:[5,1,1,""]},"sconce.monitors.base.CompositeMonitor":{add_monitor:[5,3,1,""],end_session:[5,3,1,""],start_session:[5,3,1,""],write:[5,3,1,""]},"sconce.monitors.base.Monitor":{end_session:[5,3,1,""],start_session:[5,3,1,""],write:[5,3,1,""]},"sconce.monitors.dataframe_monitor":{DataframeMonitor:[5,1,1,""]},"sconce.monitors.dataframe_monitor.DataframeMonitor":{df:[5,2,1,""],from_file:[5,4,1,""],is_blacklisted:[5,3,1,""],plot:[5,3,1,""],plot_learning_rate_survey:[5,3,1,""],save:[5,3,1,""],start_session:[5,3,1,""],write:[5,3,1,""]},"sconce.monitors.losswise_monitor":{LosswiseMonitor:[5,1,1,""]},"sconce.monitors.losswise_monitor.LosswiseMonitor":{start_session:[5,3,1,""],write:[5,3,1,""]},"sconce.monitors.ringbuffer_monitor":{RingbufferMonitor:[5,1,1,""]},"sconce.monitors.ringbuffer_monitor.RingbufferMonitor":{mean:[5,3,1,""],movement_index:[5,2,1,""],start_session:[5,3,1,""],std:[5,3,1,""],write:[5,3,1,""]},"sconce.monitors.stdout_monitor":{StdoutMonitor:[5,1,1,""]},"sconce.monitors.stdout_monitor.StdoutMonitor":{start_session:[5,3,1,""],write:[5,3,1,""]},"sconce.rate_controllers":{base:[6,0,0,"-"],constant_rate_controller:[6,0,0,"-"],cosine_rate_controller:[6,0,0,"-"],exponential_rate_controller:[6,0,0,"-"],linear_rate_controller:[6,0,0,"-"],step_rate_controller:[6,0,0,"-"],triangle_rate_controller:[6,0,0,"-"]},"sconce.rate_controllers.base":{RateController:[6,1,1,""]},"sconce.rate_controllers.base.RateController":{new_learning_rate:[6,3,1,""],start_session:[6,3,1,""]},"sconce.rate_controllers.constant_rate_controller":{ConstantRateController:[6,1,1,""]},"sconce.rate_controllers.constant_rate_controller.ConstantRateController":{new_learning_rate:[6,3,1,""],reset_monitor:[6,3,1,""],start_session:[6,3,1,""]},"sconce.rate_controllers.cosine_rate_controller":{CosineRateController:[6,1,1,""]},"sconce.rate_controllers.cosine_rate_controller.CosineRateController":{new_learning_rate:[6,3,1,""],start_session:[6,3,1,""]},"sconce.rate_controllers.exponential_rate_controller":{ExponentialRateController:[6,1,1,""]},"sconce.rate_controllers.exponential_rate_controller.ExponentialRateController":{new_learning_rate:[6,3,1,""],should_continue:[6,3,1,""],start_session:[6,3,1,""]},"sconce.rate_controllers.linear_rate_controller":{LinearRateController:[6,1,1,""]},"sconce.rate_controllers.linear_rate_controller.LinearRateController":{new_learning_rate:[6,3,1,""],should_continue:[6,3,1,""],start_session:[6,3,1,""]},"sconce.rate_controllers.step_rate_controller":{StepRateController:[6,1,1,""]},"sconce.rate_controllers.step_rate_controller.StepRateController":{new_learning_rate:[6,3,1,""],start_session:[6,3,1,""]},"sconce.rate_controllers.triangle_rate_controller":{TriangleRateController:[6,1,1,""]},"sconce.rate_controllers.triangle_rate_controller.TriangleRateController":{new_learning_rate:[6,3,1,""],start_session:[6,3,1,""]},"sconce.trainer":{Trainer:[2,1,1,""]},"sconce.trainer.Trainer":{checkpoint:[2,3,1,""],load_model_state:[2,3,1,""],multi_train:[2,3,1,""],num_trainable_parameters:[2,2,1,""],restore:[2,3,1,""],save_model_state:[2,3,1,""],survey_learning_rate:[2,3,1,""],test:[2,3,1,""],train:[2,3,1,""]},"sconce.trainers":{autoencoder_trainer:[7,0,0,"-"],classifier_trainer:[7,0,0,"-"]},"sconce.trainers.autoencoder_trainer":{AutoencoderMixin:[7,1,1,""],AutoencoderTrainer:[7,1,1,""]},"sconce.trainers.autoencoder_trainer.AutoencoderMixin":{plot_input_output_pairs:[7,3,1,""],plot_latent_space:[7,3,1,""]},"sconce.trainers.classifier_trainer":{ClassifierMixin:[7,1,1,""],ClassifierTrainer:[7,1,1,""]},"sconce.trainers.classifier_trainer.ClassifierMixin":{get_classification_accuracy:[7,3,1,""],get_confusion_matrix:[7,3,1,""],plot_confusion_matrix:[7,3,1,""],plot_samples:[7,3,1,""]},"sconce.utils":{Progbar:[2,1,1,""]},"sconce.utils.Progbar":{add:[2,3,1,""],update:[2,3,1,""]},sconce:{data_generator:[2,0,0,"-"],models:[3,0,0,"-"],monitors:[5,0,0,"-"],rate_controllers:[6,0,0,"-"],trainer:[2,0,0,"-"],trainers:[7,0,0,"-"],utils:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:classmethod"},terms:{"100x1x28x28":2,"100x3x32x32":[],"6th":2,"class":[2,3,4,5,6,7],"float":2,"function":[2,6],"int":2,"return":2,"true":[2,3,4,7],For:2,The:[2,6],These:2,__next__:2,_input:5,_output:5,abc:[5,6,7],accept:2,activ:[3,4],adaptiveaveragepooling2dlay:3,add:2,add_monitor:5,adjust:[2,6],after:[2,6],all:2,allow:2,along:2,also:2,analyz:2,ani:2,anoth:2,api_kei:5,arbitrari:2,arg:[5,6],argument:2,around:2,autoencod:7,autoencoder_train:[1,2],autoencodermixin:7,autoencodertrain:7,automat:2,averag:2,back:[2,6],bar:2,base:[1,2,3,4,7],basic_autoencod:[1,2],basic_classifi:[1,2],basicautoencod:3,basicclassifi:3,batch:2,batch_multipl:2,batch_multipli:2,batch_siz:2,befor:2,begin:[2,6],behavior:2,below:2,between:6,bia:3,blacklist:5,bool:2,both:2,cache_result:7,calcul:2,calculate_loss:[2,3],calculate_metr:[2,3],call:2,callabl:2,can:[2,6],capac:5,checkpoint:2,chosen:2,cifar100:2,cifar10:2,classifier_train:[1,2],classifiermixin:7,classifiertrain:7,classmethod:[2,3,5],complet:2,composit:2,composite_monitor:5,compositemonitor:5,consist:2,constant:6,constant_rate_control:[1,2],constantratecontrol:6,constraint:2,constructor:2,contain:2,content:1,conv_bia:3,conv_channel:3,convolution2d_lay:[2,3],convolution2dlay:4,convolutional_autoencoder_plus_perceptron:[1,2],convolutional_layer_kwarg:3,convolutionalautoencoderplusperceptron:3,convolutionallay:3,copi:2,cosin:6,cosine_rate_control:[1,2],cosineratecontrol:[2,6],cours:6,cpu:2,creat:2,cuda:2,current:2,cycl:2,cycle_length:2,cycle_multipli:2,data:[2,5,6],data_gener:[1,7],data_load:2,data_loc:2,dataframe_monitor:[1,2],dataframemonitor:[2,5],datagener:2,dataload:2,dataset:2,dataset_class:2,decod:3,deconvolutionallay:3,defin:2,depend:2,depth:3,detail:2,detect:6,determin:2,devic:2,dict:2,dictionari:2,directli:2,displai:2,dodgerblu:5,download:2,drop:6,drop_factor:6,dropout:4,dure:2,earli:2,ect:2,effect:2,els:2,encod:3,end:6,end_sess:5,endlessli:2,epoch:2,equal:2,everi:2,exampl:2,expect:2,exponenti:6,exponential_rate_control:[1,2],exponentialratecontrol:[2,6],extern:2,factor:2,fall:6,fals:[2,3,4],fashionmnist:2,few:2,fig:5,figsiz:[5,7],figure_kwarg:5,figure_width:7,file:2,filenam:[2,5],first:2,floattensor:2,follow:6,form:2,forward:[2,3,4],fraction:2,freeze_batchnorm_lay:3,from:[2,6],from_dataset:2,from_fil:5,from_pytorch:2,frome:2,fully_connected_lay:[2,3],fully_connected_layer_kwarg:3,fullyconnectedlay:4,func:[],gener:[],get_classification_accuraci:7,get_confusion_matrix:7,given:2,gpu:2,gradient:2,greater:2,has:[2,6],have:2,heatmap_kwarg:7,hidden_featur:3,hidden_s:3,histori:5,how:2,imag:2,image_channel:3,image_height:[3,7],image_width:3,in_channel:[3,4],in_featur:3,in_height:4,in_siz:4,in_width:4,includ:2,increas:2,index:[0,2],individu:2,inf:2,infer:2,inplace_activ:[3,4],input:[2,3],instanti:2,instead:2,interv:2,is_blacklist:5,iter:2,its:6,kei:[2,5],kept:2,kernel_s:[3,4],keyword:2,kwarg:[2,3,5],label:7,larg:2,latent:7,latent_s:3,later:2,layer:[0,2,3],layer_kwarg:3,learn:[2,5,6],learning_r:[5,6],learning_rate_color:5,least:2,len:2,length:2,like:2,linear_rate_control:[1,2],linearli:6,linearratecontrol:[2,6],list:2,live:2,load:2,load_model_st:2,loader:2,locat:2,longtensor:2,loop:2,loss:[2,5,6],loss_kei:6,losswise_monitor:[1,2],losswisemonitor:5,made:2,mai:2,make:2,mani:2,max_graph:5,max_learning_r:[2,6],maximum:2,mean:5,mediumseagreen:5,memori:2,metadata:5,method:2,metric:[2,5,6],metric_nam:5,min_graph:5,min_learning_r:[2,6],minimum:2,mnist:2,mode:2,model:[0,1,2,7],modifi:2,modul:[0,1],monitor:[0,1,2,7],more:2,move:6,movement_index:5,movement_kei:6,movement_threshold:6,movement_window:6,much:2,multi_train:2,multilayer_perceptron:[1,2],multilayerperceptron:3,multipl:2,multipli:2,must:2,name:[2,5],never:2,new_from_yaml_fil:3,new_from_yaml_filenam:3,new_learning_r:6,next:2,none:[2,5,6,7],note:2,num_block:3,num_categori:3,num_col:7,num_cycl:2,num_drop:6,num_epoch:2,num_sampl:[2,7],num_step:[5,6],num_trainable_paramet:2,num_work:2,number:2,object:2,occur:2,often:2,one:2,onli:2,optim:[2,7],option:2,orchestr:2,origin:2,otehr:2,other:[2,5],otherwis:2,out_channel:[3,4],out_height:4,out_siz:4,out_width:4,output:[2,3],output_featur:3,output_pad:3,output_s:3,over:[2,6],packag:[0,1],pad:[3,4],page:0,param:5,paramet:2,pass:2,path:2,perceptron:3,perceptron_featur:3,pil:2,pin:2,pin_memori:2,plot:[2,5],plot_confusion_matrix:7,plot_input_output_pair:7,plot_kwarg:5,plot_latent_spac:7,plot_learning_rate_survei:5,plot_sampl:7,practic:2,preactiv:[3,4],predict:7,predicted_label:7,preprocess:[],previou:2,previous:2,progbar:2,progbar_kwarg:5,progress:2,propag:2,put:2,pytorch:2,rate:[2,5,6],rate_control:[1,2,7],rate_controller_class:2,rate_controller_kwarg:2,rate_monitor:2,ratecontrol:[2,6],record:2,refer:[0,2],rel:2,relu:3,remain:6,represent:7,reset:2,reset_monitor:6,reshuffl:2,restor:2,restrict:2,result:7,retain:2,ringbuffer_monitor:[1,2],ringbuffermonitor:5,rise:[2,6,7],root:2,run:2,sampl:[2,7],satisfi:2,save:[2,5],save_model_st:2,scale:6,score:7,screen:2,search:0,second:2,see:2,self:2,semi:2,sent:2,session:2,set:2,shift:6,should:2,should_continu:6,shuffl:2,silent:2,simple_convolutional_autoencod:[1,2],simpleconvolutionalautoencod:3,simul:2,size:2,skip_first:5,smooth_window:5,some:6,someth:[],sort_bi:7,sourc:[2,3,4,5,6,7],specifi:2,start:2,start_sess:[5,6],state:2,stateful_metr:2,std:5,stdout_monitor:[1,2],stdoutmonitor:[2,5],step:[2,5,6],step_rate_control:[1,2],stepratecontrol:6,stop:[2,6],stop_factor:[2,6],store:2,stride:[3,4],string:2,subclass:2,submodul:1,subpackag:1,subprocess:2,support:2,survei:2,survey_learning_r:2,system:2,tag:5,take:2,target:[2,3],temporari:2,tensor:2,test:[2,5],test_color:5,test_data_gener:[2,7],test_loss:5,test_to_train_ratio:2,than:2,them:2,thi:2,thin:2,through:2,time:[2,6],titl:[5,7],tomato:5,torch:[2,3,4],torchvis:2,total:2,totensor:2,train:[2,5],trainabl:2,trainer:[0,1],training_color:5,training_data_gener:[2,7],training_loss:[5,6],transform:2,triangle_rate_control:[1,2],triangleratecontrol:6,true_label:7,tupl:2,turn:[],two:2,type:2,typic:2,underli:2,unfreeze_batchnorm_lay:3,unknown:2,until:2,updat:2,usag:2,use:2,used:2,util:1,valu:2,value_for_last_step:2,variabl:2,variou:2,verbos:2,version:2,visual:2,were:2,where:2,which:2,wide_resnet_image_classifi:[1,2],widening_factor:3,wideresnetblock_3x3:3,wideresnetgroup_3x3:3,wideresnetimageclassifi:3,width:2,with_batchnorm:[3,4],without:2,would:2,wrap:2,wrapper:2,write:5,x_in:[3,4],x_latent:3,x_out:3,y_in:3,y_out:3,yaml_fil:3,yaml_filenam:3,yield:2,you:2},titles:["Welcome to sconce\u2019s documentation!","sconce","sconce package","sconce.models package","sconce.models.layers package","sconce.monitors package","sconce.rate_controllers package","sconce.trainers package"],titleterms:{autoencoder_train:7,base:[5,6],basic_autoencod:3,basic_classifi:3,classifier_train:7,constant_rate_control:6,content:[2,3,4,5,6,7],convolution2d_lay:4,convolutional_autoencoder_plus_perceptron:3,cosine_rate_control:6,data_gener:2,dataframe_monitor:5,document:0,exponential_rate_control:6,fully_connected_lay:4,indic:0,layer:4,linear_rate_control:6,losswise_monitor:5,model:[3,4],modul:[2,3,4,5,6,7],monitor:5,multilayer_perceptron:3,packag:[2,3,4,5,6,7],rate_control:6,ringbuffer_monitor:5,sconc:[0,1,2,3,4,5,6,7],simple_convolutional_autoencod:3,stdout_monitor:5,step_rate_control:6,submodul:[2,3,4,5,6,7],subpackag:[2,3],tabl:0,trainer:[2,7],triangle_rate_control:6,util:2,welcom:0,wide_resnet_image_classifi:3}})