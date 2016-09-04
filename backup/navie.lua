require 'Monitor'

local train = require'train'

-- load data


-- construct model


-- build model
local md = nn.Sequential()
local d = p.res * p.res
local sz = {d, math.ceil(d*1.2), math.ceil(d*1.44), math.ceil(d*1.2), d }
for i = 1, #sz-1 do
  md:add(nn.Linear(sz[i], sz[i+1]))
  md:add(nn.ReLU())
end


local monitor = Monitor()


local train_config = {}

train(md, criterion, trainX, trainY, train_config)

