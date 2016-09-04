--
-- p= { train, test, trainex, testex, model, criterion}

local evaluate = function(md, cr, trainset, testset, trainexp, testexp, prefix)

 o = md:forward(testX)
    local l = cr:forward(o, testY)
    evals[#evals+1] = l
    print( string.format('%16d\t%16.2e\r\n', epoch, l) )
    
      -- plot the evaluate result
      gnuplot.epsfigure(foutput)
      gnuplot.plot(torch.Tensor(evals))
      gnuplot.plotflush()


end

local function eval_on_dataset(md, cr, X, Y)

  local O = md.forward(X)
  local L = cr.forward(O, Y)
end

return evaluate 