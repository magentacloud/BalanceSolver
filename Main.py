import json
import numpy as np
from cvxopt import matrix, solvers
from flask import Flask, request
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Sample API', description='A sample API',)


name_space = api.namespace('balance', description='Main API')

class Solver:
    def __init__(self, data):
        self.data = json.loads(data)
        self.graph = self.data['graph']
        self.matrix = None
        self.flows = None
        self.tols = None
        self.limitation_vector = None
        self.limitation_rhs = None

    def solve(self):
        self.matrix = self.create_incidence_matrix_from_graph(self.graph)
        self.tols = self.create_tols_vector(self.graph)
        self.flows = self.create_flows_vector(self.graph)
        if len(self.data['limitations']) > 0:
            self.limitation_vector = matrix(self.data['limitations']['vector'],
                                            (self.data['limitations']['limitation_count'],
                                            self.graph['edgesCount']), 'd')
            self.limitation_rhs = matrix(self.data['limitations']['rhs'],
                                         (self.data['limitations']['limitation_count'], 1), 'd')
        try:
            res = self.calculate(self.tols, self.flows, self.matrix, np.zeros(self.graph['nodeCount']),
                                 self.limitation_vector, self.limitation_rhs)
            if res is None:
                return -1
            return self.create_result_json(res)
        except ValueError:
            return -1



    def create_incidence_matrix_from_graph(self, graph):
        matrix = np.zeros((graph['nodeCount'], graph['edgesCount']))
        nodes = graph['scheme']['nodes']
        counter = 0
        for i in nodes:
            income = i['income']
            outcome = i['outcome']
            for j in income:
                matrix[counter][j - 1] = 1
            for j in outcome:
                matrix[counter][j - 1] = -1
            counter += 1
        return matrix

    def create_tols_vector(self, graph):
        vector = np.zeros((graph['edgesCount'], 1))
        edges = graph['scheme']['edges']
        counter = 0
        for edge in edges:
            vector[counter] = edge['flow'] * edge['error'] / 100
            counter += 1
        return vector

    def create_flows_vector(self, graph):
        vector = np.zeros((graph['edgesCount'], 1))
        edges = graph['scheme']['edges']
        counter = 0
        for edge in edges:
            vector[counter] = edge['flow']
            counter += 1
        return vector

    def cvxopt_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
        P = .5 * (P + P.T)  # make sure P is symmetric
        args = [matrix(P), matrix(q)]

        if A is not None:
            args.extend([matrix(A), matrix(b)])
        else:
            args.extend([None, None])
        if G is not None:
            args.extend([matrix(G), matrix(h)])
        sol = solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return sol['x']

    def calculate(self, t, x0, l, c, aeq=None, beq=None):
        n = len(t)
        h = np.eye(n)
        i = 0
        while i < n:
            h[i, i] = 1 / (t[i] ** 2)
            i += 1
        d = -h.dot(x0)

        return self.cvxopt_solve_qp(h, d, l, c, aeq, beq)

    def create_result_json(self, raw_result):
        dictionary = {}
        counter = 0
        while counter < len(raw_result):
            dictionary[f'x{counter}'] = raw_result[counter]
            counter += 1
        return json.loads(json.dumps(dictionary))

    #def is_balance_converges(self, result):
     #   for i in self.matrix:
      #      res = 0
       #     counter = 0
        #    for el in i:
         #       res += el * result[counter]
     #     counter += 1
     #       if round(res, 10) != 0:
     #           return False
       # return True

    #def is_limitations_passed(self, result):
      #  first_counter = 0
      #  limitations = np.array(self.data['limitations']['vector']).transpose()
      #  for i in limitations:
       #     res = 0
    #    counter = 0
       #     for j in i:
       #         res += j * result[counter]
       #         counter += 1
       #     if res > self.data['limitations']['rhs'][first_counter]:
     #           return False
      #      first_counter += 1
       # return True


model = api.model('Model', {
    'graph': fields.Nested(
        api.model('Graph', {
            'nodeCount': fields.Integer(),
            'edgesCount': fields.String(),
            'nodes': fields.Nested(api.model('Node', {
                'index': fields.Integer(),
                'income': fields.List(fields.Integer()),
                'outcome': fields.List(fields.Integer()),
            })),
            'edges': fields.Nested(api.model('Edge', {
                'index': fields.Integer(),
                'flow': fields.Float(),
                'error': fields.Float()
            }))
        })
    ),
    'limitation': fields.Nested(
        api.model('Limitations', {
            'rhs': fields.Integer(),
            'vector': fields.List(fields.Integer())
        })
    )
})


@name_space.route("/")
class MainClass(Resource):
    @api.expect(model)
    def post(self):
        str = request.get_data()
        solver = Solver(str)
        return solver.solve()


if __name__ == '__main__':
    app.debug = True
    app.run()
