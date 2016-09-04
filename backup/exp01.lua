-- for debug
local config = {

  output = '/Users/Jing/Dropbox/dev/depth/buffer', 
  
  -- data setting
  datadir = '/Users/Jing/Dropbox/dev/benchmarks/buffer/cache', 
  res = 16,
  nTrain = 40 * 40 * 30,
  nTest  = 5 * 40 * 30,
  
  train_examples_idx = { 1, 100, 1000},
  test_examples_idx  = {1, 100, 1000},
  
  -- training setting
  batchsz      = 128, 
  evalPeriod   = 10,
  learningRate = 0.01,
  nIter = 100
}


os.execute('mkdir -p ' ..  config.output)

ds = data.DataSource('image_gray_16', 'depth_inverse_16', 'debug')
md, criterion = model.build_model('navie', config={dinput=16*16, doutput=16*16, nhidden=3, rate=1.2})
monitor = Monitor(ds.Valid)
train(md, criterion, ds.train.X, ds.train.Y, config)

return config


