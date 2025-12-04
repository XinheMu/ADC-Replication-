Testing ADC, ADC- and ADC+ on Datasets:
Welcome to the open source code of ADC. 
This version currently contain all testing material needed for running the ADC program. 
Training material for ADC shall be uploaded a few hours later.


The current ADC program can be ran in the format
conda run -n environment python ADC_Cardest.py dataset_name ADCversion output_type unit_of_variables dimension Time_min workload_size bayes_source_attribute bayes_called_attribute bayes_assist_attribute nan_to threshold draws
The model parameters are explained as follows:
dataset_name: The dataset on which to run our experiment. Currently chosen among the values 'forest', 'power', 'higgs', 'advantage'. Note that the dataset 'modulo' is codenamed 'advantage' in our numerical experiments.
ADCversion: Choose among the values 'ADC-','ADC','ADC+'
output_type: Set to 'qerror' for the program to output and display the Q-error; set to 'sel' for the program to calculate the selectivity without calculating th Q-error.
The answer sheet found at location dataset_name+'/'+dataset_name+'_real_test.npy', eg. 'power/power_real_test.npy', is needed for output type 'qerror', but not for 'sel', as the calculation of Q-error require knowledge of the actual selectivity
unit_of_variables: a list enclosed by "" indicating the numerical precision of each attribute, used for preprocessing the query. Eg. unit_of_variables equal 1 for integral attributes, 1e-1 for attributes rounded to 1 digit decimals, 1e-2 for those rounded to 2 digit decimals, etc.
dimension: dimensionality of the dataset
Time_min: early stopping time of the diffusion model
workload_size: Total number of queries to test
bayes_source_attributes; bayes_called_attributes; bayes_assist_attributes: parameters telling the model which attributes we built the Bayesnet on
nan_to: Which number did missing values get converted to, used for the dataset 'power' which contain missing values, and whose missing values we converted to -1 in accrodance with the paper "Are We Ready for Learned Cardinality Estimtion"
threshold (optional): If output_type is set to 'qerror', all queries with an error bigger than threshold will be outputted and their index will be stored to location dataset_name+'/'+dataset_name+'_high_error_list.npy', eg. 'power/power_high_error_list.npy'
draws (optional): Number of draws for predictor-corrector Monte Carlo scheme, default number 25 balances speed with precision according to my tests, but feel free to adjust if you like.


The default code for testing the four datasets are
conda run -n environment ADC_Cardest.py higgs ADC+ qerror "[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]" 7 1/1280 10000 "[1]" "[0]" "[2]"
conda run -n environment ADC_Cardest.py power ADC+ qerror "[1e-3,1e-3,1e-2,2e-1,1,1,1]" 7 1/1280 10000 "[3]" "[0]" "[6]" -1
conda run -n environment ADC_Cardest.py forest ADC+ qerror "[1,1,1,1,1,1,1,1,1,1]" 10 1/160 10000
conda run -n environment ADC_Cardest.py advantage ADC+ qerror "[1,1,1,1,1]" 5 1/320 10000


All results are stored in the CSV file dataset_name+'/'+'Statistics_'+dataset_name+'.xlsx', eg. 'power/Statistics_power.xlsx', as a new sheet titled 'QError_'+ADCversion or one titled 'Selectivity_'+ADCversion, eg. "QError_ADC"; "Selectivity_ADC-" 
In output mode 'qerror', the five colums are: 'relseldis', the actual selectivity of queries sorted in ascending order; 'relsel', the actual selectivity of queries; 'estsel', the estimated selectivity of queries; 'Q', the q-error for each query; 'SortQ', q-error of all queries sorted in ascending order
In output mode 'sel', the two columns are:'estseldis', the estimated selectivity of queries sorted in ascending order; 'estsel', the estimated selectivity of different queries


Feel free to use new test queries if you like. In that case, to test k new queries on my model, please edit the file dataset_name+'/'+dataset_name+'_testset.csv', eg. 'power/power_testset.csv' to contain 2k rows and 'dimension' columns, with row (2k-2) containing the query lower bounds for each attribute, and row (2k-1) containing the query upper bounds for each atribute. 
Note that attributes NOT included in the "WHERE" clause STILL need to appear in the .csv file, in this case, their respective query [lower bound]/[upper bound] is simple the [lower bound of that attribute in the dataset, minus 1]/[upper bound of that attribute in the dataset, plus 1]. 
Adjust the file dataset_name+'/'+dataset_name+'_real_test.npy' along with the list of queries if you want my program to output Q-error rather than selectivity values. 


The latest version of all libraries found in the "import" commands, as of December 4 2025, are found to make the code run successfully.
If they do not due to older or drastically newer libraries, check the environment info below:


name: my_conda_env
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - binutils_impl_linux-64=2.40=h5293946_0
  - binutils_linux-64=2.40.0=hc2dff05_2
  - blas=1.0=mkl
  - bottleneck=1.4.2=py310ha9d4c09_0
  - brotli-python=1.0.9=py310h6a678d5_9
  - bzip2=1.0.8=h5eee18b_6
  - c-ares=1.19.1=h5eee18b_0
  - ca-certificates=2025.9.9=h06a4308_0
  - certifi=2025.8.3=py310h06a4308_0
  - charset-normalizer=3.3.2=pyhd3eb1b0_0
  - contourpy=1.3.1=py310hdb19cb5_0
  - cpuonly=2.0=0
  - cycler=0.11.0=pyhd3eb1b0_0
  - cyrus-sasl=2.1.28=h52b45da_1
  - expat=2.7.1=h6a678d5_0
  - ffmpeg=4.3=hf484d3e_0
  - filelock=3.17.0=py310h06a4308_0
  - fontconfig=2.14.1=h55d465d_3
  - fonttools=4.55.3=py310h5eee18b_0
  - freetype=2.13.3=h4a9f257_0
  - gcc_impl_linux-64=11.2.0=h1234567_1
  - gcc_linux-64=11.2.0=h5c386dc_2
  - giflib=5.2.2=h5eee18b_0
  - gmp=6.3.0=h6a678d5_0
  - gmpy2=2.2.1=py310h5eee18b_0
  - gnutls=3.6.15=he1e5248_0
  - gxx_impl_linux-64=11.2.0=h1234567_1
  - gxx_linux-64=11.2.0=hc2dff05_2
  - icu=73.1=h6a678d5_0
  - idna=3.7=py310h06a4308_0
  - intel-openmp=2023.1.0=hdb19cb5_46306
  - jinja2=3.1.6=py310h06a4308_0
  - joblib=1.4.2=py310h06a4308_0
  - jpeg=9e=h5eee18b_3
  - kernel-headers_linux-64=3.10.0=he073ed8_18
  - kiwisolver=1.4.8=py310h6a678d5_0
  - krb5=1.20.1=h143b758_1
  - lame=3.100=h7b6447c_0
  - lcms2=2.16=h92b89f2_1
  - ld_impl_linux-64=2.40=h12ee557_0
  - lerc=4.0.0=h6a678d5_0
  - libabseil=20250127.0=cxx17_h6a678d5_0
  - libcups=2.4.2=h2d74bed_1
  - libcurl=8.12.1=hc9e6f67_0
  - libdeflate=1.22=h5eee18b_0
  - libedit=3.1.20230828=h5eee18b_0
  - libev=4.33=h7f8727e_1
  - libffi=3.4.4=h6a678d5_1
  - libgcc-devel_linux-64=11.2.0=h1234567_1
  - libgcc-ng=11.2.0=h1234567_1
  - libgfortran-ng=11.2.0=h00389a5_1
  - libgfortran5=11.2.0=h1234567_1
  - libglib=2.78.4=hdc74915_0
  - libgomp=11.2.0=h1234567_1
  - libiconv=1.16=h5eee18b_3
  - libidn2=2.3.4=h5eee18b_0
  - libjpeg-turbo=2.0.0=h9bf148f_0
  - libnghttp2=1.57.0=h2d74bed_0
  - libpng=1.6.39=h5eee18b_0
  - libpq=17.4=hdbd6064_0
  - libprotobuf=5.29.3=h3cdef7c_1
  - libssh2=1.11.1=h251f7ec_0
  - libstdcxx-devel_linux-64=11.2.0=h1234567_1
  - libstdcxx-ng=11.2.0=h1234567_1
  - libtasn1=4.19.0=h5eee18b_0
  - libtiff=4.7.0=hde9077f_0
  - libunistring=0.9.10=h27cfd23_0
  - libuuid=1.41.5=h5eee18b_0
  - libwebp=1.3.2=h9f374a3_1
  - libwebp-base=1.3.2=h5eee18b_1
  - libxcb=1.17.0=h9b100fa_0
  - libxkbcommon=1.9.1=h69220b7_0
  - libxml2=2.13.8=hfdd30dd_0
  - lightgbm=4.6.0=py310h6a678d5_0
  - llvm-openmp=14.0.6=h9e868ea_0
  - lz4-c=1.9.4=h6a678d5_1
  - markupsafe=3.0.2=py310h5eee18b_0
  - matplotlib=3.10.0=py310h06a4308_0
  - matplotlib-base=3.10.0=py310hbfdbfaf_0
  - mkl=2023.1.0=h213fc3f_46344
  - mkl-service=2.4.0=py310h5eee18b_2
  - mkl_fft=1.3.11=py310h5eee18b_0
  - mkl_random=1.2.8=py310h1128e8f_0
  - mpc=1.3.1=h5eee18b_0
  - mpfr=4.2.1=h5eee18b_0
  - mpmath=1.3.0=py310h06a4308_0
  - mysql=8.4.0=h721767e_2
  - ncurses=6.4=h6a678d5_0
  - nettle=3.7.3=hbbd107a_1
  - networkx=3.4.2=py310h06a4308_0
  - numexpr=2.10.1=py310h3c60e43_0
  - numpy=2.0.1=py310h5f9d8c6_1
  - numpy-base=2.0.1=py310hb5e798b_1
  - openh264=2.1.1=h4ff587b_0
  - openjpeg=2.5.2=h0d4d230_1
  - openldap=2.6.4=h42fbc30_0
  - openssl=3.0.17=h5eee18b_0
  - packaging=24.2=py310h06a4308_0
  - pandas=2.2.3=py310h6a678d5_0
  - pcre2=10.42=hebb0a14_1
  - pillow=11.1.0=py310hac6e08b_1
  - pip=25.1=pyhc872135_2
  - pthread-stubs=0.3=h0ce48e5_1
  - pyparsing=3.2.0=py310h06a4308_0
  - pyqt=6.7.1=py310h6a678d5_1
  - pyqt6-sip=13.9.1=py310h5eee18b_1
  - pysocks=1.7.1=py310h06a4308_0
  - python=3.10.18=h1a3bd86_0
  - python-dateutil=2.9.0post0=py310h06a4308_2
  - python-tzdata=2025.2=pyhd3eb1b0_0
  - pytorch=2.5.1=py3.10_cpu_0
  - pytorch-mutex=1.0=cpu
  - pytz=2024.1=py310h06a4308_0
  - pyyaml=6.0.2=py310h5eee18b_0
  - qtbase=6.7.3=hdaa5aa8_0
  - qtdeclarative=6.7.3=h6a678d5_0
  - qtsvg=6.7.3=he621ea3_0
  - qttools=6.7.3=h80c7b02_0
  - qtwebchannel=6.7.3=h6a678d5_0
  - qtwebsockets=6.7.3=h6a678d5_0
  - readline=8.2=h5eee18b_0
  - requests=2.32.3=py310h06a4308_1
  - scikit-learn=1.6.1=py310h6a678d5_0
  - scipy=1.15.3=py310h525edd1_0
  - setuptools=78.1.1=py310h06a4308_0
  - sip=6.10.0=py310h6a678d5_0
  - six=1.17.0=py310h06a4308_0
  - sqlite=3.45.3=h5eee18b_0
  - sympy=1.13.3=py310h06a4308_1
  - sysroot_linux-64=2.17=h0157908_18
  - tbb=2021.8.0=hdb19cb5_0
  - threadpoolctl=3.5.0=py310h2f386ee_0
  - tk=8.6.14=h993c535_1
  - tomli=2.0.1=py310h06a4308_0
  - torchaudio=2.5.1=py310_cpu
  - torchvision=0.20.1=py310_cpu
  - tornado=6.5.1=py310h5eee18b_0
  - typing_extensions=4.12.2=py310h06a4308_0
  - tzdata=2025b=h04d1e81_0
  - unicodedata2=15.1.0=py310h5eee18b_1
  - urllib3=2.3.0=py310h06a4308_0
  - wheel=0.45.1=py310h06a4308_0
  - xcb-util=0.4.1=h5eee18b_2
  - xcb-util-cursor=0.1.5=h5eee18b_0
  - xcb-util-image=0.4.0=h5eee18b_2
  - xcb-util-renderutil=0.3.10=h5eee18b_0
  - xkeyboard-config=2.44=h5eee18b_0
  - xorg-libx11=1.8.12=h9b100fa_1
  - xorg-libxau=1.0.12=h9b100fa_0
  - xorg-libxdmcp=1.1.5=h9b100fa_0
  - xorg-xorgproto=2024.1=h5eee18b_1
  - xz=5.6.4=h5eee18b_1
  - yaml=0.2.5=h7b6447c_0
  - zlib=1.2.13=h5eee18b_1
  - zstd=1.5.6=hc292b87_0
  - pip:
      - et-xmlfile==2.0.0
      - openpyxl==3.1.5
