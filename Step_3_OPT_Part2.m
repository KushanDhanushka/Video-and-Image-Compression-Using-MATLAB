% =========================================================================

% DHANUSHKA HKK
% IMAGE AND VIDEO CODING
% Mini Project 
% Stage 3.3.2

% =========================================================================

clear all
close all
clc

% =========================================================================

% First, we extract 1st 10 frames from the desired video

% Create a VideoReader object

%---------Uncomment necessary lines when running in first time-------------

%video = VideoReader('test720p.mp4'); %(1280x720)

% Define the number of frames to extract
numframes = 10;
% frame1 => I
% frame2,3,4,...,10 => P
% This code doesn't go for any B frames.

% % Extract and save the frames as grayscale images
% for framenumber = 1:numframes
%     % Read the specified frame.
%     videoframe = readFrame(video);
% 
%     % Convert the frame to grayscale
%     grayframe = rgb2gray(videoframe);
% 
%     % Create the full output file path
%     imagename = sprintf('frame%d.jpg', framenumber);
% 
%     % Save the grayscale frame as a JPEG image
%     imwrite(grayframe,imagename,'jpg');
% end

% =========================================================================

psnrval = 48; % Keep this in between 45-50 <<<<-----------------------<<<<<<<<<<<<
fprintf('Minimum PSNR value for the reconstructed result : %s , given as an input\n',num2str(psnrval));
fprintf('\n\n');
% =========================================================================

% Initialize the cell arrays for storing macroblocks, motionvectors, inter/intra predicted frames,
% residual frame, and inter/intra invDCT for all frames.
macroblocks_frame = cell(numframes, 1);
motionvectorsofframes = cell(numframes, 1);

interpredictedframes = cell(numframes, 1); % 2-10 --> P
intrapredictedframes = cell(numframes, 1); % 2-10 --> I 

residualframes = cell(numframes, 1);

interpredictedinvDCTframes = cell(numframes,1);
intrapredictedinvDCTframes = cell(numframes,1);

% Initializing reconstructed predicted frames array (INTER PREDICTION)
repredictedframes = cell(numframes, 1);
repredictedframesh = cell(numframes, 1);
repredictedframesQP1 = cell(numframes, 1);

% Initializing reconstructed current frames array
recurrentframes = cell((numframes-1), 1);
recurrentframesh = cell((numframes-1), 1);
recurrentframesQP1 = cell((numframes-1), 1);

% Initializing inter/intra predicted frames array for Optimization
interpredictedinvDCTframesQP = cell(numframes,1);
intrapredictedinvDCTframesQP = cell(numframes,1);

% Initializing inter/intra predicted frames array for Higher Quality
interpredictedinvDCTframesh = cell(numframes,1);
intrapredictedinvDCTframesh = cell(numframes,1);

% =========================================================================

% Take Macroblocks for each frame
for framenumber = 1:numframes

    % Read the frame image (already in grayscale)
    imagename = sprintf('frame%d.jpg', framenumber);
    grayimage = imread(imagename);

    % forming macrobock cell array
    macroblocks = getmacroblocks(grayimage); % F1

    % Store the macroblocks of the current frame
    macroblocks_frame{framenumber} = macroblocks;
end

% =========================================================================

% ***************** TAKE INTRA PREDICTION ON I FRAME(1) *****************
% NOT DIFERRENTIAL CODING
% SIMPLE NEIGHBORING PREDICTION IS APPLIED   

mblocksframe1 = macroblocks_frame{1};
% Get the size of the mblocksframe1 cell array
[cellh, cellw] = size(mblocksframe1);
% Initialize Intra Predicted Macroblocks cell array on frame 1.
ipblocksframe1 = cell(cellh, cellw);

% Loop over the macroblocks, performing Intra Prediction on each one
for i = 1:cellh
    for j = 1:cellw
        mblockframe1 = mblocksframe1{i, j};
        predictedblock = getintraprediction(mblockframe1);
        ipblocksframe1{i,j} = predictedblock;
    end
end
macroblocks_frame{1} = ipblocksframe1; % storing in macroblocksarray

% Consider the frame1 (I)
interpredictedframes{1} = macroblocks_frame{1};
intrapredictedframes{1} = macroblocks_frame{1};
% No residual for I frames
residualframes{1} = zeros(size(macroblocks_frame{1}));

% =========================================================================

% ********** TAKE INTRA PREDICTION ON P FRAME(2,3,...,10) **************

for framenumber = 2:numframes % Skip 1st frame
    mblocksframe = macroblocks_frame{framenumber};
    % Get the size of the mblocksframe{framenumber} cell array
    [cellh, cellw] = size(mblocksframe);
    % Initialize Intra Predicted Macroblocks cell array on frame{framenumber}.
    ipblocksframe = cell(cellh, cellw);
    
    % Loop over the macroblocks, performing Intra Prediction on each one
    for i = 1:cellh
        for j = 1:cellw
            mblockframe = mblocksframe{i, j};
            predictedblock = getintraprediction(mblockframe);
            ipblocksframe{i,j} = predictedblock;
        end
    end
    intrapredictedframes{framenumber} = ipblocksframe; % storing in predictedmacroblocksarray
end

% =========================================================================

% *********** TAKE INTER PREDICTION ON P FRAME(2,3,...,10) *************

% Now we need to find MV s and Residuals on other remaining frames.
for framenumber = 2:numframes % Skip 1st frame

    % Get the MVs
    prevmacroblocks = macroblocks_frame{framenumber-1};
    motionvectors = getmotionvector(prevmacroblocks, macroblocks_frame{framenumber});
    % store MVs for each frame in another cell array
    motionvectorsofframes{framenumber} = motionvectors;

    % Get the predictedmacroblocks by motion compensation (Compensated image)
    interpredictedmacroblocks = getpredictedmacroblocks(prevmacroblocks, motionvectors);
    % Store the predicted macroblocks for the current frame
    interpredictedframes{framenumber} = interpredictedmacroblocks;
    
    % Initialize the residualframe cell array
    residualframe = cell(size(macroblocks_frame{framenumber}));
    
    % Loop over macroblocks_frame{framenumber}
    for i = 1:numel(macroblocks_frame{framenumber})
        % Calculate the residual for the current macroblock
        residualframe{i} = macroblocks_frame{framenumber}{i} - interpredictedmacroblocks{i};
    end

    % Store the residual frame for the current frame
    residualframes{framenumber} = residualframe;
end

% =========================================================================

% REMAINING TASKS 

% ******************************* For FRAME 1 *****************************

% Forward Transform
% No taking any residual, transmission directly encoded I frame
% Forward Transform on predictedmacroblocks cell array for I
DCTblocksf1 = getDCTonMB(interpredictedframes{1});

% =========================================================================

% First we encode the frame on High Quality
%for high quality
quality ='high';
quantizedblocksf1h = getquantized(DCTblocksf1, quality);
% Entropy Encoding
fileIDf1h = sprintf('encodeddata_frame1h.txt');
[codedictionaryf1h,bitcountf1h] = getentropycoding(quantizedblocksf1h, fileIDf1h);
% Entropy Decoding
decodedblocksf1h = getentropydecoding(codedictionaryf1h, fileIDf1h);
% Inverse Quantization
invquantblocksf1h = getinversequantized(decodedblocksf1h,quality);
% Inverse DCT
invDCTblocksf1h = getinverseDCT(invquantblocksf1h);
% storing the Inv.DCT frame
interpredictedinvDCTframesh{1} = invDCTblocksf1h;
% Reconstruction
invDCTblocksf1h = interpredictedinvDCTframesh{1};
reconf1h = getreconimage(invDCTblocksf1h);
% Calculate PSNR values
grayimageori = imread('frame1.jpg');
psnrvlf1 = psnr(reconf1h,grayimageori);

% =========================== OPTIMIZATION ================================ 

if psnrval < psnrvlf1 % We can reduce the quality
    step_size = 0.1;
    psnrv = psnrvlf1;
    QP = 0.2+step_size ; % to begin with just lower than higher quality

    while psnrv >= psnrval

        % To get the quantized cell array after forward transforming
        quantizedmacroblocksadp = getquantizedmodified(DCTblocksf1, QP);

        % Entropy Coding
        fileIDf1QP = sprintf('encodeddata_frame1QP.txt');
        [codedictionaryf1QP,bitcountf1QP] = getentropycoding(quantizedmacroblocksadp, fileIDf1QP);

        % Entropy Decoding
        decodedblocksf1QP = getentropydecoding(codedictionaryf1QP, fileIDf1QP);
        % Inverse Quantization
        invquantblocksf1QP = getinversequantizedmodified(decodedblocksf1QP,QP);
        % Inverse DCT
        invDCTblocksf1QP = getinverseDCT(invquantblocksf1QP);
        % storing the Inv.DCT frame
        interpredictedinvDCTframesQP{1} = invDCTblocksf1QP;

        % Reconstruction
        invDCTblocksf1QP = interpredictedinvDCTframesQP{1};
        reconf1QP = getreconimage(invDCTblocksf1QP);
        % Calculate PSNR values
        grayimageori = imread('frame1.jpg');
        psnrvlQP = psnr(reconf1QP,grayimageori);

        if psnrvlQP > psnrval
            psnrv = psnrvlQP;
            % Entropy Coding
            fileIDf1f = sprintf('encodeddata_frame1final.txt');
            [codedictionaryf1f,bitcountf1f] = getentropycoding(quantizedmacroblocksadp, fileIDf1f);
            QPf = QP;
            QP = QP + step_size;            
        else
            break;
        end
    end
    fprintf('Achieved QP for frame1 : %s\n',num2str(QPf));
    fprintf('\n');

else % We go with high quality - Human Eye detects this as a high quality frame
    fprintf('We go with predefined high quality \n');
    QPf = 0.2; % medium level quantization matrix divided by 5
    codedictionaryf1f = codedictionaryf1h;
    fileIDf1f = fileIDf1h;
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~ TRANSMISSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Entropy Decoding
decodedblocksf1f = getentropydecoding(codedictionaryf1f, fileIDf1f);
% Inverse Quantization
invquantblocksf1f = getinversequantizedmodified(decodedblocksf1f,QPf);
% Inverse DCT
invDCTblocksf1f = getinverseDCT(invquantblocksf1f);
% storing the Inv.DCT frames
interpredictedinvDCTframes{1} = invDCTblocksf1f;
interpredictedinvDCTframesQP{1} = invDCTblocksf1f;

% ==================== FRAMES RECONSTRUCTION ==============================

invDCTblocksf1f = interpredictedinvDCTframes{1};
reconf1f = getreconimage(invDCTblocksf1f);
% Calculate PSNR values
grayimageori = imread('frame1.jpg');
psnrvlf1f = psnr(reconf1f,grayimageori);

% ========================== DISPLAYING ===================================

fprintf('Achieved PSNR for frame1 : %s\n', num2str(psnrvlf1f));
fprintf('\n');  % This will print a blank line
figure(1);
imshow(reconf1f);
title('Reconstructed frame 1','FontSize',11);
% Save the frame as a JPEG image
imagename = sprintf('Reconstructed frame 1.jpg');
imwrite(reconf1f, imagename, 'jpg');


% ************************ For FRAMES 2 -10 *******************************

for framenumber = 2:numframes
    
    % Initializing The Inputs
    interpredictedmacroblocks = interpredictedframes{framenumber};
    residualmacroblocks = residualframes{framenumber};
    intrapredictedmacroblocks = intrapredictedframes{framenumber};
    
    % Forward Transform
    DCTblocksinterpr = getDCTonMB(residualmacroblocks); % Forward Transform on residualmacroblocks cell array
    DCTblocksintrapr = getDCTonMB(intrapredictedmacroblocks); % Forward Transform on intrapredictedmacroblocks cell array


%  #################### FOR INTER PREDICTED FRAMES ########################

    % First we encode the frame on HIGH QUALITY
    %for high quality
    quality ='high';
    quantizedblocksinterprh = getquantized(DCTblocksinterpr, quality);
    % Entropy Encoding
    fileIDinterprh = sprintf('%s_encodeddata_frameh%d.txt', 'interpr', framenumber);
    [codedictionaryinterprh,bitcountinterprh] = getentropycoding(quantizedblocksinterprh, fileIDinterprh);
    % Entropy Decoding
    decodedblocksinterprh = getentropydecoding(codedictionaryinterprh, fileIDinterprh);
    % Inverse Quantization
    invquantblocksinterprh = getinversequantized(decodedblocksinterprh,quality);
    % Inverse DCT
    invDCTblocksinterprh = getinverseDCT(invquantblocksinterprh);
    % storing the Inv.DCT frame
    interpredictedinvDCTframesh{framenumber} = invDCTblocksinterprh;

    % Reconstruction
    [recurrentframeh,repredictedframesh,recurrentframesh] = getreconimageinterpr(interpredictedinvDCTframesh, framenumber,recurrentframesh,motionvectorsofframes,repredictedframesh);   
    % Calculate PSNR values
    imageori = sprintf('frame%d.jpg', framenumber);
    grayimageori = imread(imageori);
    psnrvl1h = psnr(recurrentframeh,grayimageori);
    
    %------------------------- OPTIMIZATION ------------------------------- 

    if psnrval < psnrvl1h % We can reduce the quality
        step_size = 0.1;
        psnrv = psnrvl1h;
        QP1 = 0.2+step_size ; % to begin with just lower than higher quality

        while psnrv >= psnrval

            % To get the quantized cell array after forward transforming
            quantizedmacroblocksadp1 = getquantizedmodified(DCTblocksinterpr, QP1);

            % Entropy Coding
            fileIDQP1 = sprintf('%s_encodeddata_frameQP1%d.txt', 'interpr', framenumber);
            [codedictionaryQP1,bitcountQP1] = getentropycoding(quantizedmacroblocksadp1, fileIDQP1);

            % Entropy Decoding
            decodedblocksQP1 = getentropydecoding(codedictionaryQP1, fileIDQP1);
            % Inverse Quantization
            invquantblocksQP1 = getinversequantizedmodified(decodedblocksQP1,QP1);
            % Inverse DCT
            invDCTblocksQP1 = getinverseDCT(invquantblocksQP1);
            % storing the Inv.DCT frame
            interpredictedinvDCTframesQP{framenumber} = invDCTblocksQP1;

            % Reconstruction
            [recurrentframeQP1,repredictedframesQP1,recurrentframesQP1] = getreconimageinterpr(interpredictedinvDCTframesQP, framenumber,recurrentframesQP1,motionvectorsofframes,repredictedframesQP1);
            % Calculate PSNR values
            imageori = sprintf('frame%d.jpg', framenumber);
            grayimageori = imread(imageori);
            psnrvlQP1 = psnr(recurrentframeQP1,grayimageori);

            if psnrvlQP1 > psnrval
                psnrv = psnrvlQP1;
                % Entropy Coding
                fileIDinterpr = sprintf('%s_encodeddata_frame%dfinal.txt', 'interpr', framenumber);
                [codedictionaryinterpr,bitcountinterpr] = getentropycoding(quantizedmacroblocksadp1, fileIDinterpr);
                QP1f = QP1;
                QP1 = QP1 + step_size;
            else
                break;
            end
        end

    else % We go with high quality - Human Eye detects this as a high quality frame
        fprintf('We go with predefined high quality \n');
        QP1f = 0.2; % medium level quantization matrix divided by 5
        codedictionaryinterpr = codedictionaryinterprh;
        fileIDinterpr = fileIDinterprh;
        bitcountinterpr = bitcountinterprh;
    end

    % Entropy Decoding before transmittting
    decodedblocksinterpr = getentropydecoding(codedictionaryinterpr, fileIDinterpr);
    % Inverse Quantization
    invquantblocksinterpr = getinversequantizedmodified(decodedblocksinterpr,QP1f);
    % Inverse DCT
    invDCTblocksinterpr = getinverseDCT(invquantblocksinterpr);
    % storing the Inv.DCT frame
    interpredictedinvDCTframes{framenumber} = invDCTblocksinterpr;
    interpredictedinvDCTframesQP{framenumber} = invDCTblocksinterpr;
    interpredictedinvDCTframesh{framenumber} = invDCTblocksinterpr;
    % Reconstruction before transmittting
    [recurrentframe,repredictedframes,recurrentframes] = getreconimageinterpr(interpredictedinvDCTframes, framenumber,recurrentframes,motionvectorsofframes,repredictedframes);
    repredictedframesQP1{framenumber} = repredictedframes{framenumber};
    recurrentframesQP1{framenumber-1} = recurrentframes{framenumber-1};
    repredictedframesh{framenumber} = repredictedframes{framenumber};
    recurrentframesh{framenumber-1} = recurrentframes{framenumber-1};

%  #################### FOR INTRA PREDICTED FRAMES ########################

    % First we encode the frame on High Quality
    %for high quality
    quality ='high';
    quantizedblocksintraprh = getquantized(DCTblocksintrapr, quality);
    % Entropy Encoding
    fileIDintraprh = sprintf('%s_encodeddata_frame%dh.txt', 'intrapr', framenumber);
    [codedictionaryintraprh,bitcountintraprh] = getentropycoding(quantizedblocksintraprh, fileIDintraprh);
    % Entropy Decoding
    decodedblocksintraprh = getentropydecoding(codedictionaryintraprh, fileIDintraprh);
    % Inverse Quantization
    invquantblocksintraprh = getinversequantized(decodedblocksintraprh,quality);
    % Inverse DCT
    invDCTblocksintraprh = getinverseDCT(invquantblocksintraprh);
    % storing the Inv.DCT frame
    intrapredictedinvDCTframesh{framenumber} = invDCTblocksintraprh;
    % Reconstruction
    invDCTblocksh = intrapredictedinvDCTframesh{framenumber};
    reframeh = getreconimage(invDCTblocksh);    
    % Calculate PSNR values
    imageori = sprintf('frame%d.jpg', framenumber);
    grayimageori = imread(imageori);
    psnrvl2h = psnr(reframeh,grayimageori);

    %------------------------- OPTIMIZATION ------------------------------- 

    if psnrval < psnrvl2h % We can reduce the quality
        step_size = 0.1;
        psnrv = psnrvl2h;
        QP2 = 0.2+step_size ; % to begin with just lower than higher quality

        while psnrv >= psnrval

            % To get the quantized cell array after forward transforming
            quantizedmacroblocksadp2 = getquantizedmodified(DCTblocksintrapr, QP2);

            % Entropy Coding
            fileIDQP2 = sprintf('%s_encodeddata_frameQP2%d.txt', 'intrapr', framenumber);
            [codedictionaryQP2,bitcountQP2] = getentropycoding(quantizedmacroblocksadp2, fileIDQP2);

            % Entropy Decoding
            decodedblocksQP2 = getentropydecoding(codedictionaryQP2, fileIDQP2);
            % Inverse Quantization
            invquantblocksQP2 = getinversequantizedmodified(decodedblocksQP2,QP2);
            % Inverse DCT
            invDCTblocksQP2 = getinverseDCT(invquantblocksQP2);
            % storing the Inv.DCT frame
            intrapredictedinvDCTframesQP{framenumber} = invDCTblocksQP2;

            % Reconstrunction
            invDCTblocksQP2 = intrapredictedinvDCTframesQP{framenumber};
            reframeQP2 = getreconimage(invDCTblocksQP2);
            % Calculate PSNR values
            imageori = sprintf('frame%d.jpg', framenumber);
            grayimageori = imread(imageori);
            psnrvlQP2 = psnr(reframeQP2,grayimageori);

            if psnrvlQP2 > psnrval
                psnrv = psnrvlQP2;
                % Entropy Coding
                fileIDintrapr = sprintf('%s_encodeddata_frame%dfinal.txt', 'intrapr', framenumber);
                [codedictionaryintrapr,bitcountintrapr] = getentropycoding(quantizedmacroblocksadp2, fileIDintrapr);
                QP2f = QP2;
                QP2 = QP2 + step_size;
            else
                break;
            end
        end

    else % We go with high quality - Human Eye detects this as a high quality frame
        fprintf('We go with predefined high quality \n');
        psnrv = psnrvlf1;
        QP2f = 0.2; % medium level quantization matrix divided by 5
        codedictionaryintrapr = codedictionaryintraprh;
        fileIDintrapr = fileIDintraprh;
        bitcountintrapr = bitcountintraprh;
    end

% ============ DECISION ON WHAT KIND OF PREDICTION TO BE USED =============

    if bitcountinterpr >= bitcountintrapr

        fprintf('Best Bitrate for frame%d is given by INTRA PREDICTION',framenumber);
        fprintf('\n');
        fprintf('Achieved QP for frame%d : %s\n', framenumber, num2str(QP2f));
        fprintf('\n');

        %~~~~~~~~~~~~~~~~~~~~~~~~~~ TRANSMISSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~

        % Entropy Decoding
        decodedblocksintrapr = getentropydecoding(codedictionaryintrapr, fileIDintrapr);
        % Inverse Quantization
        invquantblocksintrapr = getinversequantizedmodified(decodedblocksintrapr,QP2f);
        % Inverse DCT
        invDCTblocksintrapr = getinverseDCT(invquantblocksintrapr);
        % storing the Inv.DCT frame
        intrapredictedinvDCTframes{framenumber} = invDCTblocksintrapr;
        intrapredictedinvDCTframesQP{framenumber} = invDCTblocksintrapr;

        % ================= FRAMES RECONSTRUCTION =========================

        invDCTblocksintrapr = intrapredictedinvDCTframesQP{framenumber};
        reframe = getreconimage(invDCTblocksintrapr);

        % Display the current frame
        figure;
        imshow(reframe);
        title(sprintf('Reconstructed Frame %d', framenumber));

        % Save the grayscale frame as a JPEG image
        imagename = sprintf('Reconstructed Frame%d.jpg', framenumber);
        imwrite(reframe, imagename, 'jpg');

        % Calculate PSNR values
        imageori = sprintf('frame%d.jpg', framenumber);
        grayimageori = imread(imageori);
        psnrvl2 = psnr(reframe,grayimageori);
        fprintf('Achieved PSNR for frame%d : %s\n', framenumber, num2str(psnrvl2));
        fprintf('\n');  % This will print a blank line
    
    else

        fprintf('Best Bitrate for frame%d is given by INTER PREDICTION',framenumber);
        fprintf('\n');
        fprintf('Achieved QP for frame%d : %s\n', framenumber, num2str(QP1f));
        fprintf('\n');

        %~~~~~~~~~~~~~~~~~~~~~~~~~~ TRANSMISSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~

        % Entropy Decoding 
        decodedblocksinterpr = getentropydecoding(codedictionaryinterpr, fileIDinterpr);
        % Inverse Quantization
        invquantblocksinterpr = getinversequantizedmodified(decodedblocksinterpr,QP1f);
        % Inverse DCT
        invDCTblocksinterpr = getinverseDCT(invquantblocksinterpr);
        % storing the Inv.DCT frame
        interpredictedinvDCTframes{framenumber} = invDCTblocksinterpr;
        
        % ================= FRAMES RECONSTRUCTION =========================

        % Reconstruction
        [recurrentframe,repredictedframes,recurrentframes] = getreconimageinterpr(interpredictedinvDCTframes, framenumber,recurrentframes,motionvectorsofframes,repredictedframes);
        % Display the current frame
        figure;
        imshow(recurrentframe);
        title(sprintf('Reconstructed Frame %d', framenumber));

        % Save the grayscale frame as a JPEG image
        imagename = sprintf('Reconstructed Frame%d.jpg', framenumber);
        imwrite(recurrentframe, imagename, 'jpg');

        % Calculate PSNR values
        imageori = sprintf('frame%d.jpg', framenumber);
        grayimageori = imread(imageori);
        psnrvl1 = psnr(recurrentframe,grayimageori);
        fprintf('Achieved PSNR for frame%d : %s\n', framenumber, num2str(psnrvl1));
        fprintf('\n');  % This will print a blank line
    end
end


% =========================================================================
% =========================================================================
% =========================================================================


%---------------------------------------------------------
% FUNCTIONS
%---------------------------------------------------------

% F1 = Form Macroblocks
function macroblocks = getmacroblocks(img)
    % Get the size of the image
    [rows, cols] = size(img);
    % Initialize a cell array to hold the macroblocks
    % Floor to avoid division errors
    macroblocks = cell(floor(rows/8), floor(cols/8));
    % Loop over the image, extracting 8x8 blocks
    for r = 1:8:rows % with steps of 8
        for c = 1:8:cols % with steps of 8
            % Extract the 8x8 block
            block = img(r:min(r+7,rows), c:min(c+7,cols));
            % min to ensure that the block does not extend beyond the edge of the image.
            padblock = zeros(8,8); % Zero Padding
            % Padding process for ensure 8x8 MBs
            padblock(1:size(block,1),1:size(block,2)) = block;
            % Store the block in the cell array
            macroblocks{floor(r/8) + 1, floor(c/8) + 1} = padblock; 
            % floor to ensures that the indices are always integers.
        end
    end
end

%---------------------------------------------------------
% F2 = Get DCT on Macroblocks
function dctblocks = getDCTonMB(mb)

    % Get the size of the MB cell array
    [cellh, cellw] = size(mb);
    % Initialize DCT cell array 
    dctblocks = cell(cellh, cellw);

    % Loop over the macroblocks, performing the DCT on each one
    for i = 1:cellh
        for j = 1:cellw
            % Perform the DCT on the macroblock from the MB cell array
            dctblock = dct2(mb{i, j});
            % Store the DCT block in the DCT cell array
            dctblocks{i, j} = dctblock;
        end
    end
end

%---------------------------------------------------------
% F3 = Quantization matrix selection among 3 levels (Low, Medium, High) and
% on required bitrate
function qmat = Quantizationmatrixselector(quality)
% Define the quantization matrices for low, medium, and high quality and
% adaptive cases.
% base quantization matrix = medium quality quantization matrix

% Defined using JPEG standards.

    Q_low = [80 55 50 80 120 200 255 305
             60 60 70 95 130 290 300 275
             70 65 80 120 200 285 345 280
             70 85 110 145 255 435 400 310
             90 110 185 280 340 545 515 385
             120 175 275 320 405 520 565 460
             245 320 390 435 515 605 600 505
             360 460 475 490 560 500 515 495];

    Q_medium = [16 11 10 16 24 40 51 61
                12 12 14 19 26 58 60 55
                14 13 16 24 40 57 69 56
                14 17 22 29 51 87 80 62
                18 22 37 56 68 109 103 77
                24 35 55 64 81 104 113 92
                49 64 78 87 103 121 120 101
                72 92 95 98 112 100 103 99];    
    
    Q_high = [3 2 2 3 5 8 10 12
              2 2 3 4 5 12 12 11
              3 3 3 5 8 11 14 11
              3 3 4 6 10 17 16 12
              4 4 7 11 14 22 21 15
              5 7 11 13 16 21 23 18
              10 13 16 17 21 24 24 20
              14 18 19 20 22 20 21 20];

    % Choose the quantization matrix based on the specified quality
    switch quality
        case 'low'
            qmat = Q_low;
        case 'medium'
             qmat = Q_medium;
        case 'high'
             qmat = Q_high;
        otherwise
            error('Invalid quality level. Choose from ''low'', ''medium'', or ''high''.');
    end
end

%---------------------------------------------------------
% F4 = Quantization of DCT cell array
function qtblocks = getquantized(DCTblocks,quality)
    % Select the quantization matrix w.r.t the quality
    qmat = Quantizationmatrixselector(quality);
    % Get the size of the DCT cell array
    [cellh, cellw] = size(DCTblocks);
    % Initialize Quantized cell array 
    qtblocks = cell(cellh, cellw);

    % Loop over the DCTblocks, performing QUANTIZATION on each one
    for i = 1:cellh
        for j = 1:cellw
            % Take one block from the cell array
            DCTblock = DCTblocks{i, j};
            % Quantize the DCT block
            quantizedblock = round(DCTblock ./ qmat);
            % Store the DCT block in the DCT cell array
            qtblocks{i, j} = quantizedblock;
        end
    end 
end

%---------------------------------------------------------
% F5 = Entropy coding with Huffmann algorithm and generate the Codebook
function [codedict,bitcount] = getentropycoding(quantizedblocks, filename)

    % Get the size of the quantizedBlocks cell array
    [cellh, cellw] = size(quantizedblocks);

    % Initialize cell arrays to hold the encoded data and code dictionaries
    encodeddata = cell(cellh, cellw);
    codedictionarydata = cell(cellh, cellw);

    % Open the text file for writing
    fileid = fopen(filename, 'w');
    bitcount = 0; % To count total number of bits in the encoded data

    % Loop over the quantizedblocks cell array
    for i = 1:cellh
        for j = 1:cellw

            % Get the current quantized block
            quantizedBlock = quantizedblocks{i, j};
            % Reshape the quantized block into a 1D vector
            quantizedVector = quantizedBlock(:);

            % Generate Huffman code dictionary
            % Find the unique symbols in the quantized vector
            uniquesymbols = unique(quantizedVector);
            % Count the number of unique symbols
            numuniquesymbols = numel(uniquesymbols); 
            % Error correction for cases which there is only one unique
            % symbol remains.
            if numuniquesymbols == 1
                uniquesymbols = [uniquesymbols 1]; % Add a new element to uniqueSymbols
                numuniquesymbols = numel(uniquesymbols);
            end % Now The value for n_ary is less than the number of distinct symbols.
            % Calculate the probabilities of each unique symbol in the quantized vector
            probabilities = histcounts(quantizedVector, numuniquesymbols) / numel(quantizedVector);
            probabilities = probabilities / sum(probabilities);
            % Generate a Huffman code dictionary based on the unique symbols and their probabilities
            dict = huffmandict(uniquesymbols, probabilities);
            % Perform Huffman coding
            compresseddata = huffmanenco(quantizedVector, dict);

            % count the number of bits used in compression.
            bitcount = bitcount + length(compresseddata);

            % stores the encoded vector and the code dictionary for each
            % block in separate cell arrays which are initialized at the
            % beginign of this function.
            encodeddata{i, j} = compresseddata;
            codedictionarydata{i, j} = dict;

            % Convert the encoded data to a string format
            encodedastring = num2str(compresseddata');
            % Write the encoded string to the text file
            fprintf(fileid, '%s\n', encodedastring);

        end
    end

    % Close the text file
    fclose(fileid);
    % output code dictionary data (for decoder)
    codedict = codedictionarydata;
end

%---------------------------------------------------------
% F6 = Entropy decoding with Huffmann algorithm
function decodedblocks = getentropydecoding(codedict, filename)
    % Open the text file for reading
    fileID = fopen(filename, 'r');

    % Get the size of the codedict cell array
    [cellh, cellw] = size(codedict);
    % Initialize cell array to store the decoded data
    decodedblocks = cell(cellh, cellw);

    % Loop over the codedict cell array
    for i = 1:cellh
        for j = 1:cellw
            % Read the encoded data from the text file as one line from the
            % file at one time.
            encodedstring = fgetl(fileID);
            % Convert the encoded string to a numeric vector(col)
            encodeddata = str2num(encodedstring');
            % Get the current code dictionary
            codedictblock = codedict{i,j};
            % Decode the encoded vector using Huffman decoding
            decodededblock = huffmandeco(encodeddata, codedictblock);
            % Reshape the quantized vector back to the original block shape
            decodededblock1D = reshape(decodededblock, 8, 8);
            % Store the decoded quantized block
            decodedblocks{i, j} = decodededblock1D;
        end
    end

    % Close the text file
    fclose(fileID);
end

%---------------------------------------------------------
% F7 = Inverse Quantization
function iqtblocks = getinversequantized(decodedcells,quality)
    % get the relevant quantization matrix used in encoder
    qmat = Quantizationmatrixselector(quality);
    % Get the size of the decodeddata cell array
    [cellh, cellw] = size(decodedcells);
    % Initialize cell array to store the inverse quantized data
    iqtblocks = cell(cellh, cellw);
    
    % Loop over the decodedcells cell array
    for i = 1:cellh
        for j = 1:cellw
           % Take one block from the decodeddata cell array
            decblock = decodedcells{i, j};
            % inverse quantized the decoded block using relevant
            % Quantization matrix
            iqtblock = decblock .* qmat;
            % store the inverse quantized block
            iqtblocks{i, j} = iqtblock;
        end
    end
end

%---------------------------------------------------------
% F8 = Inverse DCT 
function iDCTblocks = getinverseDCT(iqtblocks)

    % Get the size of the inv.quantized data cell array
    [cellh, cellw] = size(iqtblocks);
    % Initialize cell array to store the inv.DCT blocks
    iDCTblocks = cell(cellh, cellw);
    
    % Loop over the iqtblocks cell array
    for i = 1:cellh
        for j = 1:cellw
            % Get the current block from the input blocks
            iqtblock = iqtblocks{i, j};
            
            % Apply IDCT to the current block
            iDCTblock = idct2(iqtblock);
    
            % Store the transformed block
            iDCTblocks{i, j} = iDCTblock;
            
        end
    end
end

%---------------------------------------------------------
% F9 = Image Reconstruction
function reconimage = getreconimage(iDCTblocks)

    % Get the size of the inv.DCT cell array
    [cellh, cellw] = size(iDCTblocks);
    % Get the number of rows of each block
    rownumblock = size(iDCTblocks{1, 1}, 1);
    % Get the number of columns of each block
    colnumblock = size(iDCTblocks{1, 1}, 2);

    % Image size
    imageh = cellh * rownumblock; % Hieght
    imagew = cellw * colnumblock; % Width
    
    % Initialize the matrix for the image
    reconimage = uint8(zeros(imageh, imagew));
    
    % Loop over the iDCTblocks cell array
    for i = 1:cellh
        for j = 1:cellw
            % Take on block from iDCTblocks cell array.
            idctblock = iDCTblocks{i, j};

            % Row indices in the reconimage matrix where the current idctblock will be placed.
            rowIndices = (i - 1) * rownumblock + 1 : i * rownumblock; 
            % Column indices in the reconimage matrix where the current idctblock will be placed.
            colIndices = (j - 1) * colnumblock + 1 : j * colnumblock; 
            % The idctblock is placed into the appropriate position in the reconimage matrix.
            reconimage(rowIndices, colIndices) = idctblock; 
        end
    end
end

%---------------------------------------------------------
% F10 = Intra Prediction
function predictedblock = getintraprediction(original_block)
   
    % Get the size of the original_block
    [h, w] = size(original_block);
    % Initialize cell array to store the inv.DCT blocks
    predictedblock = zeros(h, w);
    
    % DC prediction - PIXELWISE PREDICTION ON NEIGHBORING PIXELS
    for i = 1:h
        for j = 1:w
            if (i >= 1) && (i <= h) && (j >= 1) && (j <= w)
                % use the average of the available neighboring pixels
                
                % for corners
                if i == 1 && j == 1
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i, j+1), original_block(i+1, j), original_block(i+1, j+1)]);
                elseif i == h && j == 1
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i-1, j), original_block(i, j+1), original_block(i-1, j+1)]);
                elseif i == 1 && j == w
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i, j-1), original_block(i+1, j), original_block(i+1, j-1)]);
                elseif i == h && j == w
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i, j-1), original_block(i-1, j), original_block(i-1, j-1)]);                        
                % for borders
                elseif i == 1 && j ~= 1 && j ~= w
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i, j-1), original_block(i, j+1), original_block(i+1, j), original_block(i+1, j-1), original_block(i+1, j+1)]);
                elseif i == h && j ~= 1 && j ~= w
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i, j-1), original_block(i, j+1), original_block(i-1, j), original_block(i-1, j-1), original_block(i-1, j+1)]);    
                elseif j == 1 && i ~= 1 && i ~= h
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i-1, j), original_block(i+1, j), original_block(i, j+1), original_block(i-1, j+1), original_block(i+1, j+1)]);
                elseif j == w && i ~= 1 && i ~= h
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i-1, j), original_block(i+1, j), original_block(i, j-1), original_block(i-1, j-1), original_block(i+1, j-1)]);
                % for body (remaining)
                else
                    predictedblock(i,j) = mean([original_block(i, j), original_block(i-1, j), original_block(i+1, j), original_block(i, j-1), original_block(i-1, j-1), original_block(i+1, j-1), original_block(i, j+1), original_block(i-1, j+1), original_block(i+1, j+1)]);
                end
            end
        end
    end
end

%---------------------------------------------------------
% F11 = Motion Vector Calculation
% ASSUME THAT THERE IS NO LARGER SUDDEN MOTION CHANGES BETWEEN FRAMES.
function motionvector = getmotionvector(previousmbs, currentmbs)

    % Get the size of the currentmacroblocks cell array
    [cellh,cellw] = size(currentmbs);
    % initialize the motion vector matrix
    % 3D matrix with dimensions cellh x cellw x 2.
    % 3rd dimension stores the motion in the x and y directions.
    motionvector = zeros(cellh, cellw, 2);
    
    % Loop over the currentmacroblocks cell array
    for i = 1:cellh
        for j = 1:cellw
            % get the current block (only one block)
            currentblock = currentmbs{i, j};
            
            % within a one single block_-_-_-
            % set the initial minimum mse and motion vector
            sad_min = Inf; % Large number to get the reduction from iterations.
            x_motion = 0;
            y_motion = 0;
            
            % loop to get best matching macroblock in the previous frame within
            % a neighborhood of the current position.
            % Go with a Hierarchical search (Not Logrithmic)
            for m = -4:4 % 8 positions
                for n = -4:4 % 8 positions
            % Total number of positions = 8x8 = 64 
            % Therefore search area is much high within neighbor blocks
                    % compute the candidate position in the previous frame
                    candidate_i = i + x_motion + m; % To get closer to best match value
                    candidate_j = j + y_motion + n; % without get closing bestmatch can be missed.
                    
                    % check if the candidate position is within the
                    % macroblock region.
                    if (candidate_i >= 1) && (candidate_i <= cellh) && (candidate_j >= 1) && (candidate_j <= cellw)
                        
                        % get the candidate block from the previous frame
                        previousblock = previousmbs{candidate_i, candidate_j};
                        
                        % compute the SAD between the current and previous(candidate) blocks
                       sad = sum(abs(currentblock(:) - previousblock(:)));
                        
                        % update the motion vector when the SAD is smaller
                        if sad < sad_min
                            sad_min = sad;
                            x_motion = x_motion + m;
                            y_motion = y_motion + n;
                            % we can get the minumum SAD and MV of that of
                            % the block.
                        end
                    end
                end
            end
            
            % store the final motion vector for the current block
            motionvector(i, j, 1) = x_motion;
            motionvector(i, j, 2) = y_motion;
        end
    end
end

%---------------------------------------------------------
% F12 = Motion Compensation
function predictedmacroblocks = getpredictedmacroblocks(prevmacroblocks, motionvector)

    % Get the size of the motionvector cell array
    [cellh,cellw,celld] = size(motionvector); % celld is not needed

    % initialize predictedmacroblocks cell array.
    predictedmacroblocks = cell(cellh, cellw);
    
    % Loop over the motionvector cell array
    for i = 1:cellh
        for j = 1:cellw

            % get the motion vector for the current macroblock
            x_motion = motionvector(i, j, 1);
            y_motion = motionvector(i, j, 2);
            
            % get the position of the referenced macroblock on previous frame
            ref_i = i + x_motion;
            ref_j = j + y_motion;
            
            % check if the referenced macroblock is within the macroblock region
            if (ref_i >= 1) && (ref_i <= cellh) && (ref_j >= 1) && (ref_j <= cellw)

                % get the referenced macroblock from the previous frame
                ref_macroblock = prevmacroblocks{ref_i, ref_j};
                
                % store that in predictedmacroblocks array
                predictedmacroblocks{i, j} = ref_macroblock;
            else
                % if the referenced macroblock is outside of boundary, set the predictedmacroblock to Null 
                predictedmacroblocks{i, j} = zeros(size(prevmacroblocks{1, 1}));

            end
        end
    end
end

%---------------------------------------------------------
% F13 = Reconstruction from iDCT frames and Mvs
function reconstructedcurrentframe = reconstructcurrentframe(repredictedmacroblocks, residualmacroblocks)

    % Get the size of the repredictedmacroblocks cell array
    [cellh, cellw] = size(repredictedmacroblocks);
    % Initialize the reconstructed current frame
    frameh = cellh * 8;
    framew = cellw * 8;
    reconstructedcurrentframe = zeros(frameh, framew);
    
    % Loop over repredictedmacroblocks cell array
    for i = 1:cellh
        for j = 1:cellw

            % Calculate the exact positions of the current macroblock in the frame
            startX = i*8 -7;
            startY = j*8 -7;
            endX = startX + 7;
            endY = startY + 7;

            % Get one reconstructed predictedmacroblock
            repredictedmacroblock = repredictedmacroblocks{i, j};

            % Get one residualmacroblock 
            residualmacroblock = residualmacroblocks{i, j};

            % Reconstruct the current macroblock by adding the predicted and residual macroblocks
            currentmacroblock = repredictedmacroblock + residualmacroblock;

            % Assign the current macroblock to the corresponding position in the frame
            reconstructedcurrentframe(startX:endX, startY:endY) = currentmacroblock;
        end
    end
end

%---------------------------------------------------------
% F14 = Getting the bitlength of image
function bitlength = getbitlengthimg(image)
    [rows,cols] = size(image); % Get the dimensions of the image
    bitlength = rows*cols* 8; % Calculate the total number of bits
    % Since grayscale, 8 bits are allocated to one pixel position
end

%---------------------------------------------------------
% F15 = Optimization for Quantization Process
function qtblocks = getquantizedmodified(DCTblocks, QP)
    % Select the quantization matrix w.r.t QP.
    % base quantization matrix = medium quality quantization matrix
    % Defined using JPEG standards.
    Q_medium = [16 11 10 16 24 40 51 61
        12 12 14 19 26 58 60 55
        14 13 16 24 40 57 69 56
        14 17 22 29 51 87 80 62
        18 22 37 56 68 109 103 77
        24 35 55 64 81 104 113 92
        49 64 78 87 103 121 120 101
        72 92 95 98 112 100 103 99];
    % Calculate the quantization matrix based on QP value   
    qmat = round(Q_medium*QP);

    % Get the size of the DCT cell array
    [cellh, cellw] = size(DCTblocks);
    % Initialize Quantized cell array 
    qtblocks = cell(cellh, cellw);

    % Loop over the DCTblocks, performing QUANTIZATION on each one
    for i = 1:cellh
        for j = 1:cellw
            % Take one block from the cell array
            DCTblock = DCTblocks{i, j};
            % Quantize the DCT block
            quantizedblock = round(DCTblock ./ qmat);
            % Store the DCT block in the DCT cell array
            qtblocks{i, j} = quantizedblock;
        end
    end 
end

%---------------------------------------------------------
% F16 = Inverse Quantization (Optimized)
function iqtblocks = getinversequantizedmodified(decodedcells,QP)
    % get the relevant quantization matrix used in encoder
    % Select the quantization matrix w.r.t QP.
    % base quantization matrix = medium quality quantization matrix
    % Defined using JPEG standards.
    Q_medium = [16 11 10 16 24 40 51 61
        12 12 14 19 26 58 60 55
        14 13 16 24 40 57 69 56
        14 17 22 29 51 87 80 62
        18 22 37 56 68 109 103 77
        24 35 55 64 81 104 113 92
        49 64 78 87 103 121 120 101
        72 92 95 98 112 100 103 99];
    % Calculate the quantization matrix based on QP value   
    qmat = round(Q_medium*QP);

    % Get the size of the decodeddata cell array
    [cellh, cellw] = size(decodedcells);
    % Initialize cell array to store the inverse quantized data
    iqtblocks = cell(cellh, cellw);
    
    % Loop over the decodedcells cell array
    for i = 1:cellh
        for j = 1:cellw
           % Take one block from the decodeddata cell array
            decblock = decodedcells{i, j};
            % inverse quantized the decoded block using relevant
            % Quantization matrix
            iqtblock = decblock .* qmat;
            % store the inverse quantized block
            iqtblocks{i, j} = iqtblock;
        end
    end
end

%---------------------------------------------------------
% F17 = Quantization Optimization
function [codedictionary, fileID, QP]= getoptimzedQP(framenumber,DCTblocks,inputstring,bitrate,bitlength,tgtcr)
 % To get the quantized cell array after forward transforming
    % Input one quality from 'low', 'medium', and 'high'
    
    %for low quality
    quality ='low';
    quantizedblocksl = getquantized(DCTblocks, quality);
    % Entropy Encoding
    fileIDlow = sprintf('%s_encodeddata_framelow%d.txt', inputstring, framenumber);
    [codedictionarylow,bitcountlow] = getentropycoding(quantizedblocksl, fileIDlow);
    fprintf('Bitcount for low quality for %s frame%d : %s\n', inputstring, framenumber, num2str(bitcountlow));
    
    % for medium quality
    quality ='medium';
    quantizedblocksm = getquantized(DCTblocks, quality);
    % Entropy Encoding
    fileIDmed = sprintf('%s_encodeddata_framemed%d.txt', inputstring, framenumber);
    [codedictionarymed,bitcountmed] = getentropycoding(quantizedblocksm, fileIDmed);
    fprintf('Bitcount for medium quality for %s frame%d : %s\n', inputstring, framenumber, num2str(bitcountmed));
    
    %for high quality
    quality ='high';
    quantizedblocksh = getquantized(DCTblocks, quality);
    % Entropy Encoding
    fileIDhigh = sprintf('%s_encodeddata_framehigh%d.txt', inputstring, framenumber);
    [codedictionaryhigh,bitcounthigh] = getentropycoding(quantizedblocksh, fileIDhigh);
    fprintf('Bitcount for high quality for %s frame%d : %s\n', inputstring, framenumber, num2str(bitcounthigh));
    fprintf('\n');  % This will print a blank line

    %----------------------------------------------------------------------
    % Quantization with Optimization analysis to find the best QP
    if tgtcr >= 1 % check whether it's needed to compress in adaptive manner
        bitcountcomp = bitrate;
    
        if (bitrate > bitcountlow) && (bitrate < bitcountmed) % need to lower the quality
            QP = 1; % begining with medium
            while bitcountcomp >= bitrate
                % To get the quantized cell array after forward transforming
                quantizedmacroblocksadp = getquantizedmodified(DCTblocks, QP);
                % Entropy Coding
                fileID = sprintf('%s_encodeddata_frame%d.txt', inputstring, framenumber);
                [codedictionary,bitcount] = getentropycoding(quantizedmacroblocksadp, fileID);
                bitcountcomp = bitcount;
                % Display the total number of bits in the encoded data
                disp(['Bitcount: ', num2str(bitcountcomp)]);
                if bitcountcomp >= bitrate
                    if ((bitcountcomp-bitrate)>((bitcountmed-bitrate)*0.1)) && ((bitcountmed-bitrate)>(bitrate/10))
                        step_size = 1;  % larger step size until get closer to target
                        QP = QP + step_size;
                    else
                        step_size = 0.1;  % smaller step size to finer result
                        QP = QP + step_size;
                    end  
                end
            end
        end
    
        if (bitrate > bitcountmed) && (bitrate < bitcounthigh) % can be increase the quality for optimal result
            QP = 1; % begining with medium
            while bitcountcomp <= bitrate
                % To get the quantized cell array after forward transforming
                quantizedmacroblocksadp = getquantizedmodified(DCTblocks, QP);
                % Entropy Coding
                fileIDAtemp ='encodedbitstreamadptemp.txt';
                [codedictionaryadptemp,bitcountadptemp] = getentropycoding(quantizedmacroblocksadp, fileIDAtemp);
                if bitcountadptemp < bitrate
                    bitcountcomp = bitcountadptemp;
                    % Display the total number of bits in the encoded data
                    disp(['Bitcount: ', num2str(bitcountcomp)]);
                    % Entropy Coding
                    fileID =sprintf('%s_encodeddata_frame%d.txt', inputstring, framenumber);
                    [codedictionary,bitcount] = getentropycoding(quantizedmacroblocksadp, fileID);
                    bitcountcomp = bitcount;
                    step_size = (bitcountcomp - bitrate) / bitrate;  % Make the step size proportional to the difference
                    QP = QP + step_size;
                else
                    break 
                end
            end
        end
    
        if bitrate == bitcountmed
            QP =1; % For medium quality
            codedictionary = codedictionarymed;
            fileID = fileIDmed;
            bitcountcomp = bitcountmed;
            disp('Bitrate allows transmission on previously defined medium quality quantization');
        end

        if bitrate >= bitcounthigh % Higher quality quantization is okay with the bitrate
            QP = 0.2; % For high quality
            codedictionary = codedictionaryhigh;
            fileID = fileIDhigh;
            bitcountcomp = bitcounthigh;
            disp('Bitrate allows transmission on previously defined high quality quantization');
        end

        if bitrate <= bitcountlow % Limit the compression due to human vision limitations
            QP = 5; % For low quality
            codedictionary = codedictionarylow;
            fileID = fileIDlow;
            bitcountcomp = bitcountlow;
            disp('Bitrate can not be achieved with keeping understandable content. So, limit the compression for predefined low quality quantization.');
        end

    
    else % If no need of any compression, use high quality
        disp('No need to adapt quantization since high quality compression of the original frame can be transmitted through the channel ')
       QP = 0.2; % For high quality
       codedictionary = codedictionaryhigh;
       fileID = fileIDhigh;
       bitcountcomp = bitcounthigh;
       disp('Bitrate allows transmission on previously defined high quality quantization');
    end
    
    % Display the CRs and QPs
    crnew = bitlength/bitcountcomp;
    fprintf('\n');  % This will print a blank line
    fprintf('Achieved CR for %s frame%d : %s\n',inputstring, framenumber, num2str(crnew));
    fprintf('Achieved QP for %s frame%d : %s\n',inputstring, framenumber, num2str(QP));
    fprintf('\n\n');  % This will print a blank line

end

%---------------------------------------------------------
% F18 = Reconstrution frames for inter predicted frames
function [recurrentframe,repredictedframes,recurrentframes] = getreconimageinterpr(interpredictedinvDCTframes, framenumber,recurrentframes,motionvectorsofframes,repredictedframes)
    % FOR INTER PREDICTION----------------------------------------------
    %Taking inverse DCT performed frame
    invDCTblocksinterpr = interpredictedinvDCTframes{framenumber};

    if framenumber == 2 % reference frame is I (1st) frame
        prevframe = interpredictedinvDCTframes{framenumber-1};
    else % For remaining frames, we have to recreate the frames
        prevframe = recurrentframes{framenumber-2};
    end

    % Get the predictedmacroblocks by motion compensation for reconstructed
    % frames
    motionvectorsf = motionvectorsofframes{framenumber};
    repredictedmacroblocks = getpredictedmacroblocks(prevframe, motionvectorsf);

    % Store the repredicted macroblocks for the current frame
    repredictedframes{framenumber} = repredictedmacroblocks;

    % Reconstruct the current frame from the recon.predicted and inv.DCT of residual macroblocks
    recurrentframe = reconstructcurrentframe(repredictedmacroblocks, invDCTblocksinterpr);
    
    % Convert grayscale pixel values to the appropriate range for display
    recurrentframe = uint8(recurrentframe);
    % Get macroblock cell array of that frame
    recurrentframemacroblocks = getmacroblocks(recurrentframe);
    % Store in frames array
    recurrentframes{framenumber-1} = recurrentframemacroblocks;
end

