"""A version of agentnet LSTM cell with output gate layer order swapped as in danet"""
import lasagne
from lasagne import init
from lasagne.layers import DenseLayer, NonlinearityLayer

from agentnet.utils.format import check_list
from agentnet.utils.layers import clip_grads, add, mul
from agentnet.memory.gate import GateLayer




def WrongLSTMCell(prev_cell,
             prev_out,
             input_or_inputs=tuple(),
             num_units=None,
             peepholes=True,
             weight_init=init.Normal(),
             bias_init=init.Constant(),
             peepholes_W_init=init.Normal(),
             forgetgate_nonlinearity=lasagne.nonlinearities.sigmoid,
             inputgate_nonlinearity=lasagne.nonlinearities.sigmoid,
             outputgate_nonlinearity=lasagne.nonlinearities.sigmoid,
             cell_nonlinearity=lasagne.nonlinearities.tanh,
             output_nonlinearity=lasagne.nonlinearities.tanh,
             name='lstm',
             grad_clipping=5.,
             ):


    """

    Implements a one-step gated recurrent unit (GRU) with arbitrary number of units.


    :param prev_cell: input that denotes previous state (shape must be (None, n_units) )
    :type prev_cell: lasagne.layers.Layer
    :param input_or_inputs: a single layer or a list/tuple of layers that go as inputs
    :type input_or_inputs: lasagne.layers.Layer or list of such

    :param num_units: how many recurrent cells to use. None means "as in prev_state"
    :type num_units: int

    :param peepholes: If True, the LSTM uses peephole connections.
        When False, peepholes_W_init are ignored.
    :type peepholes: bool

    :param bias_init: either a lasagne initializer to use for every gate weights
                        or a list of 4 initializers for  [input gate, forget gate, cell, output gate]

    :param weight_init: either a lasagne initializer to use for every gate weights:
        or a list of two initializers,
        - first used for all weights from hidden -> <all>_gate and cell
        - second used for all weights from input(s) -> <all>_gate weights and cell
        or a list of two objects elements,
        - second list is hidden -> input gate, forget gate, cell, output gate,
        - second list of lists where list[i][0,1,2] = input[i] -> [input_gate, forget gate, cell,output gate ]

    :param peepholes_W_init: either a lasagne initializer or a list of 3 initializers for
                        [input_gate, forget gate,output gate ] weights. If peepholes=False, this is ignored.
                        
    :param <any>_nonlinearity: which nonlinearity to use for a particular gate

    :param grad_clipping: maximum gradient absolute value. 0 or None means "no clipping"


    :returns: a tuple of (new_cell,new_output) layers
    :rtype: (lasagne.layers.Layer,lasagne.layers.Layer)


    for developers:
        Works by stacking other lasagne layers;
        is a function mock, not actual class.

    """

    assert len(prev_cell.output_shape) == 2
    # if required, infer num_units
    if num_units is None:
        num_units = prev_cell.output_shape[1]
    # else check it
    assert num_units == prev_cell.output_shape[1]


    # gates and cell (before nonlinearities)

    gates = GateLayer([prev_out] + check_list(input_or_inputs),
                      [num_units] * 4,
                      channel_names=["to_ingate", "to_forgetgate", "to_cell", "to_outgate"],
                      gate_nonlinearities=None,
                      bias_init=bias_init,
                      weight_init=weight_init,
                      name=name or "")

    ingate, forgetgate, cell_input, outputgate = gates.values()




    # clip grads #1
    if grad_clipping:
        ingate, forgetgate, cell_input, outputgate = [clip_grads(lyr, grad_clipping) for lyr in
                                                     [ingate, forgetgate, cell_input, outputgate]]

    if peepholes:
        # cast bias init to a list
        peepholes_W_init = check_list(peepholes_W_init)
        assert len(peepholes_W_init) in (1, 3)
        if len(peepholes_W_init) == 1:
            peepholes_W_init *= 3
        W_cell_to_ingate_init,W_cell_to_forgetgate_init= peepholes_W_init[:2]

        peep_ingate = lasagne.layers.ScaleLayer(prev_cell,W_cell_to_ingate_init,shared_axes=[0,],
                                  name= (name or "") + ".W_cell_to_ingate_peephole")

        peep_forgetgate = lasagne.layers.ScaleLayer(prev_cell,W_cell_to_forgetgate_init,shared_axes=[0,],
                                  name= (name or "") + ".W_cell_to_forgetgate_peephole")


        ingate = add(ingate,peep_ingate)
        forgetgate = add(forgetgate,peep_forgetgate)




    # nonlinearities
    ingate = NonlinearityLayer(
        ingate,
        inputgate_nonlinearity,
        name=name+".inputgate"
    )
    forgetgate = NonlinearityLayer(
        forgetgate,
        forgetgate_nonlinearity,
        name=name+".forgetgate"
    )

    cell_input = NonlinearityLayer(cell_input,
                          nonlinearity=cell_nonlinearity,
                          name=name+'.cell_nonlinearity')


    # cell = input * ingate + prev_cell * forgetgate
    new_cell= add(mul(cell_input,ingate),
                  mul(prev_cell, forgetgate))

    # output gate
    if peepholes:
        W_cell_to_outgate_init = peepholes_W_init[2]

        peep_outgate = lasagne.layers.ScaleLayer(new_cell,W_cell_to_outgate_init,shared_axes=[0,],
                                  name= (name or "") + ".W_cell_to_outgate_peephole")

        outputgate = add(outputgate, peep_outgate)

    outputgate = NonlinearityLayer(
        outputgate,
        outputgate_nonlinearity,
        name=name+".outgate"
    )

    #cell output

    #!!!CHANGES START HERE!!!


    pre_output = mul(
        outputgate,
        new_cell,
        name=name+'.pre_output'
    )
    
    new_output=NonlinearityLayer(pre_output,
                                 output_nonlinearity,
                                 name=name+'.post_outgate_nonlinearity')

    #!!!CHANGES END HERE!!!


    return new_cell, new_output
