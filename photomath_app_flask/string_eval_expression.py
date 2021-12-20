class Conversion:

	def __init__(self, capacity):
		self.top = -1
		self.capacity = capacity
		self.array = []
		self.output = []
		self.precedence = {'+':1, '-':1, '*':2, '/':2, '^':3}
	
	def isEmpty(self):
		return True if self.top == -1 else False
	
	def peek(self):
		return self.array[-1]
	
	def pop(self):
		if not self.isEmpty():
			self.top -= 1
			return self.array.pop()
		else:
			return "$"

	def push(self, op):
		self.top += 1
		self.array.append(op)

	def isOperand(self, ch):
		return ch not in {"+", "-", "*", "/", "(", ")"}

	def notGreater(self, i):
		try:
			a = self.precedence[i]
			b = self.precedence[self.peek()]
			return True if a <= b else False
		except KeyError:
			return False

	def infixToPostfix(self, exp):		
		for i in exp.split(' '):
			if self.isOperand(i):
				self.output.append(i)
			
			elif i == '(':
				self.push(i)
    
			elif i == ')':
				while( (not self.isEmpty()) and
								self.peek() != '('):
					a = self.pop()
					self.output.append(a)
				if (not self.isEmpty() and self.peek() != '('):
					return -1
				else:
					self.pop()

			else:
				while(not self.isEmpty() and self.notGreater(i)):
					self.output.append(self.pop())
				self.push(i)

		while not self.isEmpty():
			self.output.append(self.pop())

		return (" ".join(self.output))
  
class evalpostfix:
	def __init__(self):
		self.stack =[]
		self.top =-1
	def pop(self):
		if self.top ==-1:
			return
		else:
			self.top-= 1
			return self.stack.pop()
	def push(self, i):
		self.top+= 1
		self.stack.append(i)

	def centralfunc(self, ab):
		for i in ab.split(' '):
			try:
				self.push(int(i))
			except ValueError:
				val1 = self.pop()
				val2 = self.pop()
				switcher ={'+':val2 + val1, '-':val2-val1, '*':val2 * val1, '/':val2 / val1, '^':val2**val1}
				self.push(switcher.get(i))
		return int(self.pop())

#imamo problem za negativne brojeve
#fixat cemo provjerom je li minus u sredini izmedu dva broja ili izmedu operatora- sto ne valja
def fix_neg_numbers(str1):
  final = str1
  for i in range(0, len(str1)-1):
    if (str1[i] == "-"):
      if (i == 0 or str1[i-2] in {"+", "-", "*", "/", "(", ")"}):
        str1 = str1[:i+1] + "&" + str1[i+2:]
  return str1.replace("&", "")

def fix_multidigit_numbers(str1):
  import copy
  orig = copy.deepcopy(str1)
  for i in range(0, len(str1)-1):
    if ((str1[i]) in  {str(x) for x in range(10)}):
      if ((str1[i+2]) in  {str(x) for x in range(10)}):
        str1 = str1[:i+1] + "&" + str1[i+2:]
  return str1.replace("&", ""), orig
