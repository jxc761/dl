% Cache all images in ols to a binary file
% Input Parameters:
%	fn_cahce  : the output file
% 	res       : resize image to size [res x res] 
% 	bgray     : convert rgb image to gray image or not 
% 				true : convert, false: not
% 	nofirst   : excluding the first frame of each trace or not. 
%               true: excluding the first frames
%               false: not excluding, i.e including all.
% 	precision : the precision of each value. 
%
% data file format:
% <image_1> <image_2> ...<image_i> ....<image_N>
% 	- <image_i> : <channel 1> [<channel 2>, <channel 3>]
%  	  - <cahnnel_j>: <col_1> <col_2> ... <col_n>
% 
% 
function cache_image(fn_cache, res)
cache_dir = fileparts(fn_cache);
if (~exist(cache_dir, 'dir') )
    mkdir(cache_dir);
end

N = 500 * 40 *30;
M = res*res;

X = zeros(M, N);  

parfor i = 1 : N
    % x = ds.loadImage(i); 			% x = H * W * C
    if mod(i, 1200) == 1
        fprintf('%d, ', i)
    end

    fn = getImageFilename(i);
    x = imread(fn);
    x = rgb2gray( mat2gray(x, [0, 65535]) ); 	% convert to double and between [0, 1.0]
    x = imresize(x, [res, res], 'bilinear');
    x = x';
    X(:, i) = x(:);
     
end


% save out the file
fid = fopen(fn_cache, 'w');
fwrite(fid, X(:), 'float32');
fclose(fid);

end


function fn = getImageFilename(i)
F=30;
T=40;
S=500;
image_dir='/mnt/projects/CSE_CS_MSL88/object_level_scenes_v1/img_res64x64_sl18_iso300/config_0';

traces = [ '0_0_0_0'; '0_0_1_0'; '0_0_2_0'; '0_0_3_0'; '0_0_4_0'; ...
	        '0_1_0_0'; '0_1_1_0'; '0_1_2_0'; '0_1_3_0'; '0_1_4_0'; ...
	        '1_2_0_0'; '1_2_1_0'; '1_2_2_0'; '1_2_3_0'; '1_2_4_0'; ...
	        '1_3_0_0'; '1_3_1_0'; '1_3_2_0'; '1_3_3_0'; '1_3_4_0'; ...
	        '2_4_0_0'; '2_4_1_0'; '2_4_2_0'; '2_4_3_0'; '2_4_4_0'; ...
	        '2_5_0_0'; '2_5_1_0'; '2_5_2_0'; '2_5_3_0'; '2_5_4_0'; ...
	        '3_6_0_0'; '3_6_1_0'; '3_6_2_0'; '3_6_3_0'; '3_6_4_0'; ...
	        '3_7_0_0'; '3_7_1_0'; '3_7_2_0'; '3_7_3_0'; '3_7_4_0'; ...
        ];
[f, t, s] = ind2sub([F, T, S], i);
fn = sprintf('%s/scene_%d/%s/frame%03d.png', image_dir, s-1, traces(t, :), f-1);

end
