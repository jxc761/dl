
require 'image'
require 'ols'

local function TestLoadDataSet ()
local datadir = '/Users/Jing/Dropbox/dev/benchmarks/buffer/cache'
local outputdir='.'
local examples = {{1, 1, 1}, {1, 1, 30}, {1, 40, 1}, {1, 40, 30}, {2, 1, 1}, {2, 1, 30}, {500, 1, 1}, {500, 40, 30} };
local files = {}

local images = ols.LoadDataSet(datadir, 'image', 'gray', 16)
for i, example in ipairs(examples) do
  
  local filename = string.format('%s/image_%s.png', outputdir, table.concat(example, '_') )
  local img = images[example]
  image.save(filename, img:permute(1, 3, 2))
  
  print('check image: filename: ' .. filename)
  files[#files+1] = filename
end

local depths = ols.LoadDataSet(datadir, 'depth', 'inverse', 16)
for i, example in ipairs(examples) do
  local filename = string.format('%s/depth_%s.png', outputdir, table.concat(example, '_') )
  local img = depths[example]:clone()
  local norm_img = image.minmax({tensor=img})
  image.save(filename, norm_img:permute(1, 3, 2))
  print('check depth image: filename: ' .. filename)
  files[#files+1] = filename
end


print('Remove all output files[y/n]?')
local answer = io.read()
if answer == 'y'then
  for i=1, #files do
    os.remove(files[i])
    print('remove file: ' .. files[i] )
  end    
end
print('finished....')

end

TestLoadDataSet()

require 'OLSDataset'




