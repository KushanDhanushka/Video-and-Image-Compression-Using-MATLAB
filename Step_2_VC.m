%==========================================================================

% E/18/075 --- DHANUSHKA HKK
% EE 596 --- IMAGE AND VIDEO CODING
% Mini Project 
% Stage 3.2

%==========================================================================

clear all
close all
clc

%==========================================================================

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

%==========================================================================

% Initialize the cell arrays for storing  macroblocks, motionvectors, predicted frames,
% residual frames, inverse DCT, reconstructed predicted frames, and  reconstructed current frames
% for all frames.

macroblocks_frame = cell(numframes, 1);
motionvectorsofframes = cell(numframes, 1);
predictedframes = cell(numframes, 1);
residualframes = cell(numframes, 1);
invDCTframes = cell(numframes,1);
repredictedframes = cell(numframes, 1);
recurrentframes = cell((numframes-1), 1);

%==========================================================================

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

%==========================================================================

% TAKE INTRA PREDICTION ON I FRAME(1)
% NOT DIFERRENTIAL CODING
% SIMPLE DC PREDICTION IS APPLIED   

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
predictedframes{1} = macroblocks_frame{1};
% No residual for I frames
residualframes{1} = zeros(size(macroblocks_frame{1}));

%==========================================================================

% TAKE INTER PREDICTION ON P FRAME(2,3,...,10)

% Now we need to find MV s and Residuals on other remaining frames.
for framenumber = 2:numframes % Skip I frame

    % Get the MVs
    prevmacroblocks = macroblocks_frame{framenumber-1};
    motionvectors = getmotionvector(prevmacroblocks, macroblocks_frame{framenumber});
    % store MVs for each frame in another cell array
    motionvectorsofframes{framenumber} = motionvectors;

    % Get the predictedmacroblocks by motion compensation (Compensated image)
    predictedmacroblocks = getpredictedmacroblocks(prevmacroblocks, motionvectors);
    % Store the predicted macroblocks for the current frame
    predictedframes{framenumber} = predictedmacroblocks;
    
    % Initialize the residualframe cell array
    residualframe = cell(size(macroblocks_frame{framenumber}));
    
    % Loop over macroblocks
    for i = 1:numel(macroblocks_frame{framenumber})
        % Calculate the residual for the current macroblock
        residualframe{i} = macroblocks_frame{framenumber}{i} - predictedmacroblocks{i};
    end % fast

    % Store the residual frame for the current frame
    residualframes{framenumber} = residualframe;
end

%==========================================================================


% Preview and save the predicteds and residuals of each frame

for framenumber = 2:numframes
    predictedmacroblocks = predictedframes{framenumber};
    residualmacroblocks = residualframes{framenumber};

    % Reconstruct the frame from predicted macroblocks
    predictedframe = get_pred_resi_frames(predictedmacroblocks);
    % Convert grayscale pixel values to the appropriate range for display
    predictedframe = uint8( predictedframe);
    % Display the predicted frame
    figure();
    imshow(predictedframe);
    title(sprintf('Frame %d (Predicted)', framenumber));
    % Save the predicted frame as a JPEG image
    imagenamepr = sprintf('Predicted frame%d.jpg', framenumber);
    imwrite(predictedframe, imagenamepr, 'jpg');

    % Reconstruct the frame from residual macroblocks
    residualframe = get_pred_resi_frames(residualmacroblocks);
    % Convert grayscale pixel values to the appropriate range for display
    residualframe = uint8( residualframe);
    % Display the residual frame
    figure();
    imshow(residualframe);
    title(sprintf('Frame %d (Residual)', framenumber));    
    % Save the residual of the  frame as a JPEG image   
    imagenamers = sprintf('Residual frame%d.jpg', framenumber);
    imwrite(residualframe, imagenamers, 'jpg');
end

%==========================================================================

% NOW NEED TO FOCUS ON REMAINING TASKS (SAME TASKS AS IN IMAGE CODING)

for framenumber = 1:numframes

    predictedmacroblocks = predictedframes{framenumber};
    residualmacroblocks = residualframes{framenumber};

    if framenumber ==1 % For I frames
        % No taking any residual, transmission directly encoded I frame
        % Forward Transform on predictedmacroblocks cell array for I
        DCTblocks = getDCTonMB(predictedmacroblocks);

    else % For Non-I frames
        % Forward Transform on residualmacroblocks cell array
        DCTblocks = getDCTonMB(residualmacroblocks);
    end

    %==========================================================================

    % To get the quantized cell array after forward transforming
    % Input one quality from 'low', 'medium', and 'high'
    quality ='high';
    quantizedblocks = getquantized(DCTblocks, quality);

    %==========================================================================

    % Entropy Encoding
    fileID = sprintf('encodeddata_frame%d.txt', framenumber);
    [codedictionary,bitcount] = getentropycoding(quantizedblocks, fileID);
    
    % get the compression ratio at each frame
    imagename = sprintf('frame%d.jpg', framenumber);
    frame = imread(imagename);
    bitlength = getbitlengthimg(frame);
    compressionratio = bitlength/bitcount;
    fprintf('Compression ratio for frame%d : %s\n', framenumber, num2str(compressionratio));

    %==========================================================================

    %===========================TRANSMISSION===================================

    %==========================================================================

    % Entropy Decoding
    decodedblocks = getentropydecoding(codedictionary, fileID);

    %==========================================================================

    % Inverse Quantization
    invquantblocks = getinversequantized (decodedblocks,quality);

    %==========================================================================

    % Inverse DCT
    invDCTblocks = getinverseDCT(invquantblocks);

    %==========================================================================

    % storing the Inv.DCT frame
    invDCTframes{framenumber} = invDCTblocks;
    
end

%==========================================================================

% RECONSTRUCT THE I FRAME
invDCTblocksI = invDCTframes{1};
reIframe = getreconimage(invDCTblocksI);
repredictedframes{1} = invDCTframes{1}; % for reconstruction of remaining frames
figure(1);
imshow(reIframe);
title('Reconstructed I frame (frame 1)','FontSize',11);
% Save the frame as a JPEG image
imagename = sprintf('Reconstructed I frame_(frame 1).jpg');
imwrite(reIframe, imagename, 'jpg');
% Calculate PSNR values
grayimage1 = imread('frame1.jpg');
psnrvl1 = psnr(reIframe,grayimage1);
fprintf('\n');  % This will print a blank line
fprintf('Achieved PSNR for frame1 by Intra Prediction : %s\n', num2str(psnrvl1));

%==========================================================================

% RECONSTRUCT THE P FRAMES USING MVS AND RESIDUALS

for framenumber = 2:numframes
    
    %Taking inverse DCT performed frame
    invDCTblocks = invDCTframes{framenumber};

    if framenumber == 2 % reference frame is I (1st) frame
        prevframe = invDCTframes{framenumber-1};
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
    recurrentframe = reconstructcurrentframe(repredictedmacroblocks, invDCTblocks);
    
    % Convert grayscale pixel values to the appropriate range for display
    recurrentframe = uint8(recurrentframe);
    % Get macroblock cell array of that frame
    recurrentframemacroblocks = getmacroblocks(recurrentframe);
    % Store in frames array
    recurrentframes{framenumber-1} = recurrentframemacroblocks;
    
    % Display the current frame
    figure;
    imshow(recurrentframe);
    title(sprintf('Frame %d (Current)', framenumber));
    
    % Create the full output file path
    imageName3 = sprintf('Reconstructed Frame%d.jpg', framenumber);
  
    % Save the grayscale frame as a JPEG image
    imwrite(recurrentframe, imageName3, 'jpg');

     % Calculate PSNR values
    imageori = sprintf('frame%d.jpg', framenumber);
    grayimageori = imread(imageori);
    psnrvl1 = psnr(recurrentframe,grayimageori);
    fprintf('Achieved PSNR for frame%d by Inter Prediction : %s\n', framenumber, num2str(psnrvl1));
end



%==========================================================================
%==========================================================================
%==========================================================================



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
            for m = -4:4 % 9 positions
                for n = -4:4 % 9 positions
            % Total number of positions = 9x9 = 81
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
% F15 = Construction of the Predicted frames and Residuals - for testing
function constructedframe = get_pred_resi_frames(macroblocks)
    [cellh, cellw] = size(macroblocks);
    frameh = cellh * 8;
    framew = cellw * 8;
    constructedframe = zeros(frameh, framew);
    
    for i = 1:cellh
        for j = 1:cellw

            % Calculate the exact positions of the current macroblock in the frame
            startX = i*8 -7;
            startY = j*8 -7;
            endX = startX + 7;
            endY = startY + 7;
            
            % Retrieve the macroblock
            macroblock = macroblocks{i, j};
            
            % Place the macroblock in the frame
            constructedframe(startX:endX, startY:endY) = macroblock;
        end
    end
end