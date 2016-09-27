require 'image'
require 'torch'

local function test(filename, res) 

	local file = torch.DiskFile(filename, 'r')
	file:binary()
	local storage = file:readFloat(res*res)
	local tensor=torch.FloatTensor(storage, 1, torch.LongStorage({res, res}))
	local img = image.minmax({tensor=tensor})
	image.display(image.scale(img, 256, 256))
	file:close()

end

test('../../buffer/cache/depth_normal_16x16.cache', 16)
test('../../buffer/cache/depth_normal_32x32.cache', 32)

test('../../buffer/cache/depth_inverse_16x16.cache', 16)
test('../../buffer/cache/depth_inverse_32x32.cache', 32)

test('../../buffer/cache/depth_log_16x16.cache', 16)
test('../../buffer/cache/depth_log_32x32.cache', 32)
