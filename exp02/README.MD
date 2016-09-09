-----------------------------------------
--
-- data_size : small | full
-- split_method: scene | trace
--
-- model_struct: inc | sym
-- nhidden: 3 | 4 | 5 
-- a : 1.2, 1.5, 2, 2.4, 3 increasing 1-1
--
-- learning rate: [1e-1, 1e-6]
-- learning rate: [min, max, n, method]
-- 

The step:
1) small dataset, sym, 3, 1.2, [1e-1, 1e-6]
2) small dataset, sym, 3, 1.2, [min, max]

-- expect 
-- better than current
-- increase the model size
--
-- small dataset, sym, 3, 2.4 | 3.6, [1e-1, 1e-6]

if betterthen
  validate the 
fi