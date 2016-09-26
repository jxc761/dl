
local function measure(y1, y2) 



end

local function histc(...)
   -- get args
   local args = {...}
   local tensor = args[1] or error('Usage: histc(tensor, [nbin[, min[, max])')
   local bins = args[2] or 50
   local min = args[3] or tensor:min()
   local max = args[4] or tensor:max()
   
   local bw = (max-min)/bins 

   hist = torch.histc(tensor:double(), bins, min, max)
   centers = torch.linespace(min+0.5*bw, max-0.5*bw, bins)
   return hist, centers
end


local function saveout_expsmps(prefix, expSmpIdx, depths
end


require 'image'


local function evaluate(model, data, opts)
	-- evaluate on examples
	local predict = evaluate(model)


end


local function evaluate_on_examples(model, data, opts)
	expX = data:trainX(opts.train_exp_smp_idx)
	expY = data:trainY(opts.train_exp_smp_idx)


end

local function evaluate_on_traces(model, data, opts)
	expX = data:trainX:train( (opts.train_exp_trc_idx)
	expY = data:trainY(opts.train_exp_trc_idx)
	evaluate_on_traces(model, expX, expY, opts)

	validX = data:ValidX(opts.train_exp_trc_idx)
	validY = data:ValidY(opts.train_exp_trc_idx)
	evaluate_on_traces(model, expX, expY, opts)

	testX = data:trainX(opts.train_exp_trc_idx)
	testY = data:trainY(opts.train_exp_trc_idx)
	evaluate_on_traces(model, expX, expY, opts)
end


local function evaluate_on_traces(model, X, Y, opts)
	for t = 1, T do
		for m = 1, M do
			x = X[t][m]
			y = Y[t][m]
			local o = model:predict(x)
			local prefix = string.format('%s_t%d_s%m', opts.prefix, t, m)
			compareAll(y, o, {space=opts.space, prefix=prefix})
			local performs = measuesAll(y, o)
		end
	end
end


local function saveim(filename, im, sz)
	local img = image.minmax({tensor=im:view(sz)})
	image.save(filename, tensor)
end


local function compareAll(y, o, opts)
				if opt.space == 'log' then
				est = torch.exp(o)
			else if opt.space=='inv' then 
				est = torch.inv(o)
			else
				est = o
			end


	local logy = log(y)
	local logo = log(o)

	local invy = 1.0 / y
	local invo = 1.0 / t

	prefix = string.format('%s_log', opts.prefix)
	compare1(logy, logo, prefix)

	prefix = string.format('%s_inv', opts.prefix)
	compare1(invy, invo, prefix)

	prefix = string.format('%s_nrm', opts.prefix)
	compare1(y, o, prefix)

	return measures

end

local function compare1(groundtruth, predict, prefix)
	local d = groundtruth-predict
	local h, c = histc(d)

	-- save out the depthmap
	local fn_predict = string.format('%s_est.png', prefix)
	imsave(fn_predict, predict)

	-- save out the errormap
	local fn_errmap=string.format('%s_err.png', prefix)
	imsave(fn_errmap, d)

	-- save out the histgram
	local fn_hist = string.format('%s_hist.txt', prefix)
	local ftxt = assert( io.open(fn_hist, 'w'))
  for i = 1, h:nElement() do
	  ftxt:write( string.format('%f\t%d\r\n', h[i], c[i])
	end
	ftxt:flush()
	ftxt:close()

end


local function evaluate(model, x, y, opts)

	local o = model:predict(x)
	local f = model:forward(x, y)
	local d = y-o
	

  -- save out x, y, o
  if opts.fn_data then
  	torch.save(opts.fn_data, {x=x, y=y, o=o}, 'binary')
  end

	-- save out the predict depthmap
	if opts.fn_predict then 
		local img=o:copy()
		image.minmax({tensor=img, inplace=true})
		image.save(opts.fn_predict, img) 
	end

	-- save out the error map
	if opts.fn_errmap then 
		local img=d:copy()
		image.minmax({tensor=img, inplace=true})
		image.save(opts.fn_errmap, img) 
	end

	-- save out the histogram 
	if opts.fn_hist then

		local ftxt = assert( io.open(opts.fn_hist, 'w'))

		local hist, centers = histc(y-o)
		for i = 1, hist:nElement() do
			ftxt:write( string.format('%f\t%d\r\n', hist[i], centers[i])
		end

		ftxt:flush()
		ftxt:close()
	end

	-- save out the measurements 
	if opts.fout then 
		func = 
		mse1 = 
		mse2 =
		mse3 = 
	end

end




local function evaluate_on_examples(model, X, Y)
 	local n = 0

 	-- objective function value 
 	obj = torch.Tensor(n)
 	for i = 1, n do
 		obj[i] = model:forward(X[i], Y[i])
 	end 

  Y0 = model:predict(X)

  -- save out X, Y, Y0
  -- compute quantity measurement on them

  --


end


