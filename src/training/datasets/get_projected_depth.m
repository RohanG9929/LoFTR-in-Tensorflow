imagesDir="C:\Users\15512\Downloads\toolbox_nyu_depth_v2\office_kitchen\ppm";
depthDir="C:\Users\15512\Downloads\toolbox_nyu_depth_v2\office_kitchen\pgm";
imageFiles=dir(fullfile(imagesDir,'*.ppm'));
depthFiles=dir(fullfile(depthDir,'*.pgm'));
len=min(length(depthFiles),length(imageFiles));
for k=1:len
    st_depth=append(depthFiles(k).folder,'\',depthFiles(k).name);
    imgDepth = imread(st_depth);
    imgDepth = swapbytes(imgDepth);
    st=append(imageFiles(k).folder,'\',imageFiles(k).name);
    rgb = imread(st);
    % call the function project depth
    [depthOut, rgbOut] = project_depth_map(imgDepth,rgb);
    depth_name_mat=append('C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a\depth_projected\',' ',int2str(k),'.mat');
    save(depth_name_mat,"depthOut")
    depth_name_img=append('C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a\depth_maps\',' ',int2str(k),'.png');
    imwrite(depthOut,depth_name_img);
    image_name=append('C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a\images_undistorted\',' ',int2str(k),'.png');
    imwrite(rgbOut,image_name);
end