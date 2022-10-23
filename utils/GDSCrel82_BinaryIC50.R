calculateThresold<-function(fileName,outPDF,outFile,sd=0.2,confidenceFile="",t=0.05){
# 计算二元化阈值
# fileName 输入原始实验值数据文件名。行是样本点，列是实验指标，值为实验指标值，文件必须有表头，第一列为样本名，不同列之间由制表符分割
# outPDF 输出pdf文件名。
# outFile 输出二元化阈值文件名
# sd 扩数据时所用标准差，即所有的实验指标值都用相同的标准差
# confidenceFile 扩数据时所用的标准差文件。行是样本点，列是实验指标，。文件必须有表头，第一列为样本名，且表头和样本名均需和fileName文件保持一致
# t 用于选取二元化阈值时的累积正态分布值，默认为0.05
        require(graphics)
        library('getopt')
        library(kedd)
        library(ks)
        response<-read.table(file=fileName,header=F,sep="\t",stringsAsFactor=F);
        drugs = response[1,][-1];
        clines = response[,1][-1];
        response = response[-1,];
        response = as.matrix(response[,-1]);
        response = apply(response,2,as.numeric);
        colnames(response) = drugs;
        rownames(response) = clines;
        if(confidenceFile == ""){
            flag=FALSE;
        }else{
            flag=TRUE;
            sd = read.table(file=confidenceFile,header=F,sep="\t",stringsAsFactor=F);
            drugs1 = sd[1,][-1];
            clines1 = sd[,1][-1];
            sd = sd[-1,];
            sd = as.matrix(sd[,-1]);
            sd = apply(sd,2,as.numeric)
            colnames(sd) = drugs1;
            rownames(sd) = clines1;
            sd = sd[,as.vector(unlist(drugs))];
            sd = sd[as.vector(unlist(clines)),];
            
        }
        result<-matrix(ncol=2);
        colnames(result)<-c("drug","thresold");
        result<-result[-1,];
        pdf(outPDF,width=12);
        for(d in 1:dim(response)[2]){
            data<-response[,d];
            drug = drugs[d];
            print(paste("第",d,"个drug: ",drug,sep=""));
            pointsSet = c();
            for(i in 1:length(data)){
                if(is.na(data[i])){
                    next;
                }else{
                    mu = data[i];
                    #points = rnorm(1000, mean = mu, sd = sigma);
                    if(flag){
                        points = rnorm(1000, mean = mu, sd = sd[i,d]);
                    }else{
                        points = rnorm(1000, mean = mu, sd = sd);
                    }
                    pointsSet = c(pointsSet,points);
                   # print(i);
                } 
            }
            print(paste("data points: ",length(pointsSet),sep=""));
            layout(matrix(c(1,2),nc=2,byrow=T));
            
            sitaSingle='';
            thresold='';
            h0 <- hns(pointsSet,deriv.order=0)
            desityPoint = density(pointsSet,n=100000,kernel="gaussian",from=min(pointsSet),to=max(pointsSet),bw=h0);
            f = approxfun(desityPoint$x, desityPoint$y, yleft=0, yright=0)
            mu = desityPoint$x[which(desityPoint$y == max(desityPoint$y))];
            print(paste("均值为：",mu,sep=""));
            listQuery = seq(min(data,na.rm=T),mu,length=1000)
            candidate=c();
            h1 <- hns(pointsSet,deriv.order=1)
            dev1 = kedd::dkde(pointsSet,y=listQuery,kernel="gaussian",from=min(pointsSet),to=max(pointsSet),deriv.order=1,h=h1);
            index = which(abs(dev1$est.fx)<0.001)
            sita = dev1$eval.points[index]
            sita = sita[which(f(sita)>0)]
            if(length(sita)!=0){
                for(i in 1:length(sita)){
                   fsita = f(sita[i])
                   culsita = integrate(f, min(pointsSet),sita[i])$value
                   if(culsita>0.05 && f(sita[i])<f(mu)*0.8 && sita[i]<mu*0.9){
                      candidate=c(candidate,sita[i]);
                      #print(sita[i])
                   }
                }
            }
            rule2=TRUE;
            rule3=TRUE;
            if(length(candidate)!=0){
                print("使用规则1");
                rule2=FALSE;
                rule3=FALSE;
                sitaSingle=max(candidate);
                temp = pointsSet[which(pointsSet>=sitaSingle)];
                temp = temp[which(temp<=mu)]
                sigma = median(temp -mu)
                thresold = qnorm(0.05,mu,sigma*sigma);
            }
            if(rule2){
                h2=hns(pointsSet,deriv.order=2)
                dev2 = kedd::dkde(pointsSet,y=listQuery,kernel="gaussian",from=min(pointsSet),to=max(pointsSet),deriv.order=2,h=h2);
                index = which(abs(dev2$est.fx)<0.001);
                sita = dev2$eval.points[index];
                sita = sita[which(f(sita)>0)];
                #print(sita)
                if(length(sita)!=0){
                     #print("进入规则2并开始计算三阶导数");
                     h3=hns(pointsSet,deriv.order=3);
                     dev3 = kedd::dkde(pointsSet,y=sita,kernel="gaussian",from=min(pointsSet),to=max(pointsSet),deriv.order=3,h=h3);              
                     for(i in 1:length(sita)){
                        fsita = f(sita[i])
                        culsita = integrate(f, min(pointsSet),sita[i])$value
                        if(culsita>0.05 && f(sita[i])<f(mu)*0.8 && sita[i]<mu*0.9 && dev3$est.fx[i]>0){
                           candidate=c(candidate,sita[i]);
                          # print(sita[i]);
                        }
                     }
                 }
                 if(length(candidate)!=0){
                     print("使用规则2");
                     rule3=FALSE;
                     sitaSingle=max(candidate);
                     temp = pointsSet[which(pointsSet>=sitaSingle)];
                     temp = temp[which(temp<=mu)]
                     sigma = median(temp -mu)
                     thresold = qnorm(0.05,mu,sigma*sigma)
                 }
             }
             if(rule3){
               # print("进入规则3");
                print("使用规则3");
                sitaSingle = min(pointsSet);
                temp = pointsSet[which(pointsSet>=sitaSingle)];
                temp = temp[which(temp<=mu)];
                sigma = median(temp -mu);
                thresold = qnorm(0.05,mu,sigma*sigma);
            }
            #single  = c(drug,thresold);
            print(paste("thresold: ",thresold,sep=""))
            print(paste("sita: ",sitaSingle,sep=""))
            result<-rbind(result,as.vector(unlist(c(drug,thresold))));
            options(digits=4) 
            hist(data,breaks = 200,main=paste("Distribution of IC50 for ",drug,sep=""),xlab="IC50",col="red",border="white",ylab="frequency")
            hist(pointsSet,breaks = 200,main=paste("Upsampled distribution for ",drug,sep=""),xlab="IC50",ylab="",col="blue",border="white", yaxt="n")
            legend("topleft", c(paste("Sita:  ",sitaSingle,sep=""), "",paste("Thresold:  ",thresold,sep="")),bty = "n",)
            par(new=TRUE)
            plot(desityPoint,main="",xlab="",ylab="",axes=F,col="red",type="l",lwd=3)
        }
        dev.off();
        write.table(result,file=outFile,sep="\t",quote=F,row.names=F);       
}

calculateThresold('GDSCrel82_V1_DR_LNIC50_OMICS_expseq_DR_table.txt',
'GDSCrel82_V1_DR_LNIC50_OMICS_expseq_DR_table.pdf',
'GDSCrel82_V1_DR_binaryResponse_OMICS_expseq_DR.txt')
