clc;
clear all;

[filename ,pathname]=uigetfile('*.bmp;*.jpg','Pick an Image File');
org=imread(filename); 
[filename3,pathname]=uigetfile('*.bmp;*.jpg','select original ground truth image');
gr=imread(filename3);
[x,y,z]=size(org);
if z==1
    org=cat(3, org, org,org);
end
len=length(filename);
org= org(:,:,2);
% figure,imshow(org);
subplot(2,3,1);
imshow(org);
title('original image');
Input=org;
[r c p]   = size(Input);


% figure(3);
%s = imnoise(org, "gaussian", 0.20);
%subplot(2,3,2),
%imshow(s);
%title("Gaussian noise");
%str1=strcat('.\output\','G_noise_',filename);
%imwrite(s,str1,'jpg');
%p=fspecial('gaussian',5,5);
%It creates a two dimensional filter of the specified type.
%X=imfilter(s,p);
%subplot(2,3,3)
%It filters the multidimensional array with multidimensional filter
%imshow(X);
%title("Gausssian Filter ")
%str2=strcat('.\output\','G_filter_',filename);
%imwrite(X,str2,'jpg');


%%%  SALT AND PEPPER NOISE
s = imnoise(org, "salt & pepper", 0.9);
subplot(2, 3,2 )
imshow(s);
title("Salt and Pepper noise");
str1=strcat('.\output\','G_noise_',filename);
imwrite(s,str1,'jpg');
subplot(2,3,3)

%%%  MEDIAN FILTER
X=medfilt2(s);
imshow(X);
title("Median Filter")
str2=strcat('.\output\','G_filter_',filename);
imwrite(X,str2,'jpg');


%%%%     EXISTING METHOD CODE
% org1=im2bw(org,0.6);
org1=im2bw(X);
[m,n]=size(org1);
clusters=3; % No. of segments
[k, class, img_vect]= kmeans(org1, clusters);
%  for clust = 1:k
%  cluster = reshape(class(1:length(img_vect),clust:clust), [m,n] );
%  subplot(1,k,clust), imshow(cluster,[]), title('k-means cluster ');
%  end
%  cluster = reshape(class(1:length(img_vect),1:1), [m,n] );
%  figure,imshow(cluster,[]);
%  title('cluster-1 k-mean');


cluster = reshape(class(1:length(img_vect),2:2), [m,n] );
subplot(2,3,4);
imshow(cluster,[]);
title('Exising k-mean');
str3=strcat('.\output\','Existing_kM_',filename);
imwrite(cluster,str3,'jpg');
%%%%%%%%%%%%%%%%%%%%% END
%%%%%%%%%%%%%%%%%%%%% 
 
%%%%%%%%%%% PROPOSED K-MEAN CODE


% wd=256;
% Input=imresize(a,[256 256]);
Input=X;
[r c p]   = size(Input);
if p==3
Input= Input(:,:,2);

end

Input   =double(Input);
Length = (r*c); 
Dataset = reshape(Input,[Length,1]);
Clusters=3; %k CLUSTERS
Cluster1=zeros(Length,1);
Cluster2=zeros(Length,1);
 Cluster3=zeros(Length,1);
% Cluster4=zeros(Length,1);
% Cluster5=zeros(Length,1);
miniv = min(min(Input));
maxiv = max(max(Input));
range = maxiv - miniv;
stepv = range/Clusters;
incrval = stepv;
for i = 1:Clusters
    K(i).centroid = incrval;
    incrval = incrval + stepv;
end

update1=0;
update2=0;
 update3=0;
% update4=0;
% update5=0;

mean1=3;
mean2=3;
 mean3=3;
% mean4=2;
% mean5=2;

% while  ((mean1 ~= update1) & (mean2 ~= update2) & (mean3 ~= update3) & (mean4 ~= update4) & (mean5 ~= update5))
while  ((mean1 ~= update1) & (mean2 ~= update2)& (mean3 ~= update3) )

mean1=K(1).centroid;
mean2=K(2).centroid;
 mean3=K(3).centroid;
% mean4=K(4).centroid;
% mean5=K(5).centroid;

for i=1:Length
    for j = 1:Clusters
        temp= Dataset(i);
        difference(j) = abs(temp-K(j).centroid);
       
    end
    [y,ind]=min(difference);
    
  if ind==1
    Cluster1(i)   =temp;
end
if ind==2
    Cluster2(i)   =temp;
end
 if ind==3
     Cluster3(i)   =temp;
 end
% if ind==4
%     Cluster4(i)   =temp;
% end
% if ind==5
%     Cluster5(i)   =temp;
% end



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%UPDATE CENTROIDS
cout1=0;
cout2=0;
cout3=0;
% cout4=0;
% cout5=0;

for i=1:Length
    Load1=Cluster1(i);
    Load2=Cluster2(i);
   Load3=Cluster3(i);
%     Load4=Cluster4(i);
%     Load5=Cluster5(i);
%     
    if Load1 ~= 0
        cout1=cout1+1;
    end
    
    if Load2 ~= 0
        cout2=cout2+1;
    end
    
     if Load3 ~= 0
         cout3=cout3+1;
     end
%     
%     if Load4 ~= 0
%         cout4=cout4+1;
%     end
%     
%     if Load5 ~= 0
%         cout5=cout5+1;
%     end
end

Mean_Cluster(1)=sum(Cluster1)/cout1;
Mean_Cluster(2)=sum(Cluster2)/cout2;
 Mean_Cluster(3)=sum(Cluster3)/cout3;
% Mean_Cluster(4)=sum(Cluster4)/cout4;
% Mean_Cluster(5)=sum(Cluster5)/cout5;
% %reload
for i = 1:Clusters
    K(i).centroid = Mean_Cluster(i);

end

update1=K(1).centroid;
update2=K(2).centroid;
 update3=K(3).centroid;                                  
% update4=K(4).centroid;
% update5=K(5).centroid;

end

AA1=reshape(Cluster1,[r c]);
AA2=reshape(Cluster2,[r c]);
 AA3=reshape(Cluster3,[r c]);
% AA4=reshape(Cluster4,[r c]);
% AA5=reshape(Cluster5,[r c]);

% figure(11);
% imshow(AA1);
% figure(12);
% imshow(AA2);
% figure(13);
% imshow(AA3);
% figure(14);
% imshow(AA4);
% figure(15);
% imshow(AA5);

subplot(2,3,5);
imshow(AA3);
title('Proposed k-mean');
str4=strcat('.\output\','Proposed_KM_',filename);
imwrite(AA2,str4,'jpg');

squaredErrorImage = (double(gr) - double(AA3)) .^ 2;
% Sum the Squared Image and divide by the number of elements
% to get the Mean Squared Error.  It will be a scalar (a single number).
mse = sum(sum(squaredErrorImage)) / (r * c);
fprintf('MSE:%f\n',mse);
% Calculate PSNR (Peak Signal to Noise Ratio) from the MSE according to the formula.
PSNR_P = 10 * log10( 256^2 / mse);
% Alert user of the answer.

squaredErrorImage  = (double(gr) - double(cluster)) .^ 2;
% Sum the Squared Image and divide by the number of elements
% to get the Mean Squared Error.  It will be a scalar (a single number).
mse = sum(sum(squaredErrorImage)) / (r * c);
fprintf('MSE:%f\n',mse);
% Calculate PSNR (Peak Signal to Noise Ratio) from the MSE according to the formula.
PSNR_E= 10 * log10( 256^2 / mse);
% Alert user of the answer.


%PSNR_P = psnr(double(AA2),double(gr));
%PSNR_E = psnr(double(cluster),double(gr));



% Contour matching score for image segmentationcollapse all in page
Cscore_P = bfscore(logical(AA3),logical(gr));  
Cscore_E = bfscore(logical(cluster),logical(gr)); 

% VS-VOLUMETRIC SIMILARIY
VS_P = Evaluate(gr,AA3);
VS_E = Evaluate(gr,cluster);

% HAMMING DISTANCE
% HD = pdist2(X,Y,Distance)  Distnce='hamming'
[m,n]=size(gr);
HD1=pdist2(logical(gr),logical(AA3),'hamming');
HD_P=sum(sum(abs(HD1)))/(m*n);

HD2=pdist2(logical(gr),logical(cluster),'hamming');
HD_E=sum(sum(abs(HD2)))/(m*n);


fprintf('Parameter Values:\n');

fprintf('Between Proposed & ground truth image:\n');
fprintf('The PSNR = %.2f\n',PSNR_P);
fprintf('Contour matching:%f \n', Cscore_P);
fprintf('VS:%f \n', VS_P);

fprintf('HD:%f \n', HD_P);

fprintf('\nBetween Existing & ground truth image:\n');
fprintf('The PSNR = %.4f\n',PSNR_E);
fprintf('Contour matching:%f \n', Cscore_E);
fprintf('VS:%f \n', VS_E);
fprintf('HD:%f \n', HD_E);