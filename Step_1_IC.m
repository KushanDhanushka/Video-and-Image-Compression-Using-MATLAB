%=======================================================================

% E/18/075 --- DHANUSHKA HKK
% EE 596 --- IMAGE AND VIDEO CODING
% Mini Project 
% Stage 3.1 (3.1.1 , 3.1.2 , 3.1.3)

%=======================================================================

clear all
close all
clc

%==========================================================================

%Read the original image
originalimage = imread('Lena.png'); % 512 x 512, 462.7 KB
%originalimage = imread('test480p.jpg'); % 640 x 480, 14.5 KB @ 480p

% since I've experienced there is some videos which have a frame width non
% divisible by 8 and frame height is divisible by 8, I've applied optional
% padding paths for those kind of cases.

% Show original image
figure(1);
imshow(originalimage);
title('Original Image','FontSize',11);

%==========================================================================

%Convert the original image to grayscale
grayimage = rgb2gray(originalimage);
figure(2);
imshow(grayimage); % Show grayscaled original image
title('Gray Image','FontSize',11);
% Save this imaeg as a .jpg image.
imwrite(grayimage, 'Original.jpg');
% Get the height and width of the image (for the padding process)
[gh, gw] = size(grayimage);

%==========================================================================

macroblocks = getmacroblocks(grayimage); % @ macroblock size = 8.
% macroblocks is a 64x64 cell and each entry contains 8x8 block.

%==========================================================================

% Forward Transform
DCTmacroblocks = getDCTonMB (macroblocks);
% DCT macroblocks is a 64x64 cell and each entry contains 8x8 block.

%==========================================================================

% Quantization

% For adaptive case

% FOR 3.1.2 SECTION
% E no. = E/18/075
bitrate = (300+75)*1000; % in kbps <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% FOR 3.1.3 SECTION
%bitrate = 500*1000 ; % 500kbps <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
compressionratio = getcompressionratio(grayimage,bitrate);
% display the target compression ratio:
disp(['Target Compression Ratio: ', num2str(compressionratio)]);
factor = 1; % that means adaptive case starts with medium quality.
% factor affects only for the adaptive case!!!!

% FOR 3.1.1 SECTION
% To get the quantized cell array after forward transforming
% Input one quality from 'low', 'medium', and 'high'
quantizedmacroblockslow = getquantized(DCTmacroblocks, 'low', factor);
quantizedmacroblocksmed = getquantized(DCTmacroblocks, 'medium', factor);
quantizedmacroblockshigh = getquantized(DCTmacroblocks, 'high', factor);

%==========================================================================

% Entropy Coding
fileIDL ='encodedbitstreamlow.txt';
fileIDM ='encodedbitstreammed.txt';
fileIDH ='encodedbitstreamhigh.txt';

[codedictionarylow,bitcountlow] = getentropycoding(quantizedmacroblockslow, fileIDL);
[codedictionarymed,bitcountmed] = getentropycoding(quantizedmacroblocksmed, fileIDM);
[codedictionaryhigh,bitcounthigh] = getentropycoding(quantizedmacroblockshigh, fileIDH);

% Get the compression ratios 
compressionratioL = getcompressionratio(grayimage,bitcountlow);
compressionratioM = getcompressionratio(grayimage,bitcountmed);
compressionratioH = getcompressionratio(grayimage,bitcounthigh);

% Display the CRs
disp(['Compression Ratio at LQ: ', num2str(compressionratioL)]);
disp(['Compression Ratio at MQ: ', num2str(compressionratioM)]);
disp(['Compression Ratio at HQ: ', num2str(compressionratioH)]);

%Display the total number of bits in the encoded data
disp(['Total number of bits in the encoded data for LQ: ', num2str(bitcountlow)]);
disp(['Total number of bits in the encoded data for MQ: ', num2str(bitcountmed)]);
disp(['Total number of bits in the encoded data for HQ: ', num2str(bitcounthigh)]);

%==========================================================================

%============================TRANSMISSION==================================

%==========================================================================

% Entropy Decoding
decodedmacroblockslow = getentropydecoding(codedictionarylow, fileIDL);
decodedmacroblocksmed = getentropydecoding(codedictionarymed, fileIDM);
decodedmacroblockshigh = getentropydecoding(codedictionaryhigh, fileIDH);

%==========================================================================

% Inverse Quantization
invquantmacroblockslow = getinversequantized (decodedmacroblockslow,'low',factor);
invquantmacroblocksmed = getinversequantized (decodedmacroblocksmed,'medium',factor);
invquantmacroblockshigh = getinversequantized (decodedmacroblockshigh,'high',factor);

%==========================================================================

% Inverse DCT
invDCTmacroblockslow = getinverseDCT(invquantmacroblockslow,gw);
invDCTmacroblocksmed = getinverseDCT(invquantmacroblocksmed,gw);
invDCTmacroblockshigh = getinverseDCT(invquantmacroblockshigh,gw);

%==========================================================================

% Reconstruction image
reconstructedimagelow = getreconstructedimage(invDCTmacroblockslow,gw);
reconstructedimagemed = getreconstructedimage(invDCTmacroblocksmed,gw);
reconstructedimagehigh = getreconstructedimage(invDCTmacroblockshigh,gw);

%==========================================================================

% View the reconstructed image
figure(3);
imshow(reconstructedimagelow);
title('Reconstructed Image @ Low Quality','FontSize',11);
figure(4);
imshow(reconstructedimagemed);
title('Reconstructed Image @ Medium Quality','FontSize',11);
figure(5);
imshow(reconstructedimagehigh);
title('Reconstructed Image @ High Quality','FontSize',11);

% Save those imaegs as a .jpg image.
imwrite(reconstructedimagelow, 'Reconstructed Image_LQ.jpg');
imwrite(reconstructedimagemed, 'Reconstructed Image_MQ.jpg');
imwrite(reconstructedimagehigh, 'Reconstructed Image_HQ.jpg');

% Get the PSNR values.
psnrvall = psnr(reconstructedimagelow,grayimage);
disp('PSNR of output of reconstructed image in Low Qulaity =');
disp(psnrvall);

psnrvalm = psnr(reconstructedimagemed,grayimage);
disp('PSNR of output of reconstructed image in Medium Qulaity =');
disp(psnrvalm);

psnrvalh = psnr(reconstructedimagehigh,grayimage);
disp('PSNR of output of reconstructed image in High Qulaity =');
disp(psnrvalh);

%==========================================================================

% FOR 3.1.2 / 3.1.3 SECTIONS
% For adaptive case
if compressionratio >= 1 % check whether it's needed to compress in adaptive manner
    bitcountcomp = bitrate;
    factor = 1 ; % To begin with medium quality

    if bitrate < bitcountmed % need to lower the quality
        while bitcountcomp >= bitrate
            % To get the quantized cell array after forward transforming
            quantizedmacroblocksadp = getquantized(DCTmacroblocks, 'adaptive', factor);
            % Entropy Coding
            fileIDA ='encodedbitstreamadp.txt';
            [codedictionaryadp,bitcountadp] = getentropycoding(quantizedmacroblocksadp, fileIDA);
            bitcountcomp = bitcountadp;
            % Display the total number of bits in the encoded data
            disp(['Total number of bits in the encoded data for AQ: ', num2str(bitcountcomp)]);
            if bitcountcomp >= bitrate
                if ((bitcountcomp-bitrate)>((bitcountmed-bitrate)*0.1)) && ((bitcountmed-bitrate)>(bitrate/10))
                    step_size = 1;  % larger step size until get closer to target
                    factor = factor + step_size;
                else
                    step_size = 0.1;  % smaller step size for finer result
                    factor = factor + step_size;
                end
            end
        end
    end

    if bitrate > bitcountmed % can increase the quality for optimal result
        while bitcountcomp <= bitrate
            % To get the quantized cell array after forward transforming
            quantizedmacroblocksadp = getquantized(DCTmacroblocks, 'adaptive', factor);
            % Entropy Coding
            fileIDAtemp ='encodedbitstreamadptemp.txt';
            [codedictionaryadptemp,bitcountadptemp] = getentropycoding(quantizedmacroblocksadp, fileIDAtemp);
            if bitcountadptemp < bitrate
                bitcountcomp = bitcountadptemp;
                % Display the total number of bits in the encoded data
                disp(['Total number of bits in the encoded data: ', num2str(bitcountcomp)]);
                % Entropy Coding
                fileIDA ='encodedbitstreamadp.txt';
                [codedictionaryadp,bitcountadp] = getentropycoding(quantizedmacroblocksadp, fileIDA);
                step_size = (bitcountcomp - bitrate) / bitrate;  % Make the step size proportional to the difference
                factorori = factor;
                factor = factor + step_size;
            else
                break 
            end
        end
        factor = factorori;
    end


    if bitrate == bitcountmed % Medium quality is required.
        % To get the quantized cell array after forward transforming
        quantizedmacroblocksadp = getquantized(DCTmacroblocks, 'adaptive', factor);
        % Entropy Coding
        fileIDA ='encodedbitstreamadp.txt';
        [codedictionaryadp,bitcountadp] = getentropycoding(quantizedmacroblocksadp, fileIDA);
        bitcountcomp = bitcountadp;
        % Display the total number of bits in the encoded data
        disp(['Total number of bits in the encoded data: ', num2str(bitcountcomp)]);
    end

else % Where image is small (no need any adaptive compression, just High Quality is okay)
    disp('No need to adapt quantization since high quality compression of the original image can be transmitted through the channel ')
    % To get the quantized cell array after forward transforming
    quantizedmacroblocksadp = getquantized(DCTmacroblocks, 'high', factor);
    % Entropy Coding
    fileIDA ='encodedbitstreamadp.txt';
    [codedictionaryadp,bitcountadp] = getentropycoding(quantizedmacroblocksadp, fileIDA);
    % Display the total number of bits in the encoded data
    disp(['Total number of bits in the encoded data: ', num2str(bitcountcomp)]);
end

% Display the CRs
compressionrationew = getcompressionratio(grayimage,bitcountcomp);
disp(['Acheived Compression Ratio at AQ: ', num2str(compressionrationew)]);

%==========================================================================

%============================TRANSMISSION==================================

%==========================================================================

% Entropy Decoding
decodedmacroblocksadp = getentropydecoding(codedictionaryadp, fileIDA);

%==========================================================================

% Inverse Quantization
invquantmacroblocksadp = getinversequantized (decodedmacroblocksadp,'adaptive',factor);

%==========================================================================

% Inverse DCT
invDCTmacroblocksadp = getinverseDCT(invquantmacroblocksadp,gw);

%==========================================================================

% Reconstruction image
reconstructedimageadp = getreconstructedimage(invDCTmacroblocksadp,gw);

%==========================================================================

% View the reconstructed image
figure(6);
imshow(reconstructedimageadp);
title('Reconstructed Image @ Adaptive Quality','FontSize',11);
% Save the imaeg as a .jpg image.
imwrite(reconstructedimageadp, 'Reconstructed Image_AQ.jpg');
% Get the PSNR value.
psnrvaladp = psnr(reconstructedimageadp,grayimage);
disp('PSNR of output of reconstructed image using adaptive bitrate =');
disp(psnrvaladp);

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
            % Store the block in the cell array
            macroblocks{floor(r/8) + 1, floor(c/8) + 1} = block; 
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
% F3 = Get the Compression Ratio
function cr = getcompressionratio(image, targetbitrate)
    [w,h] = size(image);
    imagesizeinbits = w*h*8;  % for a grayscale image with 8 bits per pixel
    % Calculate the compression ratio (Assuming transmit one image per second)
    cr = imagesizeinbits / targetbitrate; 
end

%---------------------------------------------------------
% F4 = Quantization matrix selection among 3 levels (Low, Medium, High) and
% on required bitrate
function qmat = Quantizationmatrixselector(quality,factor)
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
        case 'adaptive'
            % Adjust the quantization matrix based on the compression ratio
            qmat = round(Q_medium*factor);
        otherwise
            error('Invalid quality level. Choose from ''low'', ''medium'', or ''high''.');
    end
end

%---------------------------------------------------------
% F5 = Quantization of DCT cell array
function qtblocks = getquantized(DCTblocks,quality,factor)
    % Select the quantization matrix w.r.t the quality
    qmat = Quantizationmatrixselector(quality,factor);
    % Get the size of the DCT cell array
    [cellh, cellw] = size(DCTblocks);
    % Initialize Quantized cell array 
    qtblocks = cell(cellh, cellw);

    % Loop over the DCTblocks, performing QUANTIZATION on each one
    for i = 1:cellh
        for j = 1:cellw
            % Take one block from the cell array
            DCTblockinit = DCTblocks{i, j};
            DCTblock = zeros(8,8); % Zero Padding
            % Padding process if the DCTblock width is not same as the quantization matrix width  
            DCTblock(1:size(DCTblockinit,1),1:size(DCTblockinit,2)) = DCTblockinit;
            % Quantize the DCT block
            quantizedblock = round(DCTblock ./ qmat);
            % Store the DCT block in the DCT cell array
            qtblocks{i, j} = quantizedblock;
        end
    end 
end

%---------------------------------------------------------
% F6 = Entropy coding with Huffmann algorithm and generate the Codebook
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
% F7 = Entropy decoding with Huffmann algorithm
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
% F8 = Inverse Quantization
function iqtblocks = getinversequantized(decodedcells,quality,factor)
    % get the relevant quantization matrix used in encoder
    qmat = Quantizationmatrixselector(quality,factor);
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
% F9 = Inverse DCT 
function iDCTblocks = getinverseDCT(iqtblocks,gwidth)

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
            
            % Reverse the padding process.
            mbwidth = mod(gwidth,8); 
            if mbwidth ~= 0
                if j == cellw
                    DCTblocknew = iDCTblock(:,1:mbwidth);
                    iDCTblock = reshape(DCTblocknew', [8, mbwidth]);
                end
                % Store the transformed block
                iDCTblocks{i, j} = iDCTblock;
            else
                % Store the transformed block
                iDCTblocks{i, j} = iDCTblock;
            end
        end
    end
end

%---------------------------------------------------------
% F10 = Image Reconstruction
function reconimage = getreconstructedimage(iDCTblocks,gwidth)

    % Get the size of the inv.DCT cell array
    [cellh, cellw] = size(iDCTblocks);
    % Get the number of rows of each block
    rownumblock = size(iDCTblocks{1, 1}, 1);
    % Get the number of columns of each block
    colnumblock = size(iDCTblocks{1, 1}, 2);

    % Image size
    imageh = cellh * rownumblock; % Hieght
    % Reverse the effect of padding
    mbwidth = mod(gwidth,8); 
    if mbwidth ~= 0
        imagew = (cellw-1)* colnumblock + size(iDCTblocks{1, cellw}, 2); % Width
    else
        imagew = cellw * colnumblock; % Width
    end

    % Initialize the matrix for the image
    reconimage = uint8(zeros(imageh, imagew));
    
    % Loop over the iDCTblocks cell array
    for i = 1:cellh
        for j = 1:cellw
            % Take on block from iDCTblocks cell array.
            idctblock = iDCTblocks{i, j};

            % Row indices in the reconimage matrix where the current idctblock will be placed.
            rowIndices = (i - 1) * rownumblock + 1 : i * rownumblock; 

            % Reverse the effect of padding
            mbwidth = mod(gwidth,8); 
            if mbwidth ~= 0
                if j == cellw
                    colnumblockup = size(iDCTblocks{1, cellw}, 2) ;
                    % Column indices in the reconimage matrix where the current idctblock will be placed.
                    colIndices = (j - 1) * colnumblock + 1 : (j * colnumblock + (colnumblockup-colnumblock)); 
                else
                    % Column indices in the reconimage matrix where the current idctblock will be placed.
                    colIndices = (j - 1) * colnumblock + 1 : j * colnumblock; 
                end
            else
                % Column indices in the reconimage matrix where the current idctblock will be placed.
                colIndices = (j - 1) * colnumblock + 1 : j * colnumblock; 
            end  

            % The idctblock is placed into the appropriate position in the reconimage matrix.
            reconimage(rowIndices, colIndices) = idctblock; 
        end
    end
end