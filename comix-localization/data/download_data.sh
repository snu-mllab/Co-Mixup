target=$1

if [ "$target" = "clean" ] || [ "$target" = "all" ]
then 
	echo "Download clean validation data"
	gdown https://drive.google.com/uc?id=1ln6T7leFx6eosPvjH59rY30e5t_KuOTg -O clean.tar.gz
	tar -zxvf clean.tar.gz
	rm clean.tar.gz
fi

if [ "$target" = "noise" ] || [ "$target" = "all" ]
then 
	echo "Download gaussian noise test data"
	gdown https://drive.google.com/uc?id=1lX3FeRuEGBBM-pVcWHHtiwpZR7bgnMZt -O noise.tar.gz
	tar -zxvf noise.tar.gz
	rm noise.tar.gz
fi

if [ "$target" = "rep" ] || [ "$target" = "all" ]
then 
	echo "Download replacement test data"
	gdown https://drive.google.com/uc?id=13cUpMPPx7bFHQ_Uf2Ppq5dS7un40_9gO -O rep.tar.gz
	tar -zxvf rep.tar.gz
	rm rep.tar.gz
fi


